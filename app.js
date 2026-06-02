/**
 * PulseSense — Camera-based Heart Rate Monitor
 * 
 * Uses Photoplethysmography (PPG) to detect heart rate from the camera.
 * The red channel intensity fluctuates with each heartbeat as blood volume
 * changes in the fingertip tissue.
 * 
 * Two BPM detection methods:
 * 1. Peak detection (primary) — finds local maxima in filtered signal
 * 2. Auto-correlation (fallback) — frequency domain analysis for noisy signals
 */

(function () {
    'use strict';

    // ============================================
    // Configuration
    // ============================================
    const CONFIG = {
        BUFFER_SECONDS: 10,         // Seconds of data to keep
        MIN_BPM: 40,
        MAX_BPM: 220,
        STABILIZATION_TIME: 2000,   // ms before attempting BPM
        MIN_PEAKS_FOR_BPM: 2,      // Minimum peaks needed
        SMOOTHING_WINDOW: 3,        // Moving average window size
        QUALITY_THRESHOLD: 0.08,    // Very low threshold — we try to show BPM whenever possible
        BPM_HISTORY_SIZE: 6,        // Rolling BPM readings to average
    };

    // ============================================
    // State
    // ============================================
    const state = {
        isRunning: false,
        stream: null,
        animationFrameId: null,
        startTime: 0,
        lastSampleTime: 0,
        sampleCount: 0,
        measuredFPS: 30,

        // Signal buffers
        rawSignal: [],
        filteredSignal: [],
        timestamps: [],

        // Results
        bpmHistory: [],
        allBpmReadings: [],
        currentBPM: 0,
        avgBPM: 0,
        minBPM: Infinity,
        maxBPM: 0,
        lastIBI: 0,
        signalQuality: 0,
        lastPeakTime: 0,
        detectionMethod: '',

        // Adaptive filter state
        lpX: [0, 0, 0],
        lpY: [0, 0, 0],
        hpX: [0, 0, 0],
        hpY: [0, 0, 0],

        // DC removal
        dcEstimate: 0,
    };

    // ============================================
    // DOM Elements
    // ============================================
    const DOM = {
        btnStart: document.getElementById('btn-start'),
        btnStop: document.getElementById('btn-stop'),
        instructionCard: document.getElementById('instruction-card'),
        measurementPanel: document.getElementById('measurement-panel'),
        video: document.getElementById('camera-video'),
        canvas: document.getElementById('camera-canvas'),
        waveformCanvas: document.getElementById('waveform-canvas'),
        bpmValue: document.getElementById('bpm-value'),
        bpmLabel: document.getElementById('bpm-label'),
        bpmHeart: document.getElementById('bpm-heart'),
        sqValue: document.getElementById('sq-value'),
        sqBarFill: document.getElementById('sq-bar-fill'),
        statAvg: document.getElementById('stat-avg'),
        statMin: document.getElementById('stat-min'),
        statMax: document.getElementById('stat-max'),
        statIBI: document.getElementById('stat-ibi'),
        waveformTime: document.getElementById('waveform-time'),
    };

    const ctx = DOM.canvas.getContext('2d', { willReadFrequently: true });
    const waveCtx = DOM.waveformCanvas.getContext('2d');

    // ============================================
    // Camera Access
    // ============================================
    async function startCamera() {
        try {
            // Try rear camera first (for phones)
            let constraints = {
                video: {
                    facingMode: { ideal: 'environment' },
                    width: { ideal: 320 },
                    height: { ideal: 240 },
                }
            };

            try {
                state.stream = await navigator.mediaDevices.getUserMedia(constraints);
            } catch (e) {
                // Fallback to any camera (laptops)
                constraints = { video: { width: { ideal: 320 }, height: { ideal: 240 } } };
                state.stream = await navigator.mediaDevices.getUserMedia(constraints);
            }

            DOM.video.srcObject = state.stream;

            await new Promise((resolve) => {
                DOM.video.onloadedmetadata = () => {
                    DOM.video.play();
                    resolve();
                };
            });

            // Try to enable torch/flash
            const track = state.stream.getVideoTracks()[0];
            try {
                const capabilities = track.getCapabilities ? track.getCapabilities() : {};
                if (capabilities.torch) {
                    await track.applyConstraints({ advanced: [{ torch: true }] });
                    console.log('Torch enabled');
                }
            } catch (e) {
                console.log('Torch not available');
            }

            // Set canvas size
            DOM.canvas.width = DOM.video.videoWidth || 320;
            DOM.canvas.height = DOM.video.videoHeight || 240;

            return true;
        } catch (err) {
            console.error('Camera error:', err);
            alert('Camera access denied. Please allow camera permissions and reload.');
            return false;
        }
    }

    function stopCamera() {
        if (state.stream) {
            state.stream.getTracks().forEach(t => t.stop());
            state.stream = null;
        }
        DOM.video.srcObject = null;
    }

    // ============================================
    // Signal Extraction
    // ============================================
    function extractChannels() {
        if (!DOM.canvas.width || !DOM.canvas.height) return { red: 0, green: 0, blue: 0 };

        ctx.drawImage(DOM.video, 0, 0, DOM.canvas.width, DOM.canvas.height);
        const imageData = ctx.getImageData(0, 0, DOM.canvas.width, DOM.canvas.height);
        const pixels = imageData.data;

        let redSum = 0, greenSum = 0, blueSum = 0, count = 0;

        const w = DOM.canvas.width;
        const h = DOM.canvas.height;
        // Sample the center 60% of the frame
        const x0 = Math.floor(w * 0.2);
        const x1 = Math.floor(w * 0.8);
        const y0 = Math.floor(h * 0.2);
        const y1 = Math.floor(h * 0.8);

        // Step by 2 for performance on larger frames
        const step = w > 200 ? 2 : 1;

        for (let y = y0; y < y1; y += step) {
            for (let x = x0; x < x1; x += step) {
                const i = (y * w + x) * 4;
                redSum += pixels[i];
                greenSum += pixels[i + 1];
                blueSum += pixels[i + 2];
                count++;
            }
        }

        return {
            red: count > 0 ? redSum / count : 0,
            green: count > 0 ? greenSum / count : 0,
            blue: count > 0 ? blueSum / count : 0,
        };
    }

    // ============================================
    // Signal Quality Assessment
    // ============================================
    function assessSignalQuality(channels) {
        const { red, green, blue } = channels;
        const total = red + green + blue;
        if (total < 1) return 0;

        let quality = 0;

        // 1. Red dominance (finger with flash: red > 50%)
        const redRatio = red / total;
        if (redRatio > 0.36) {
            quality += Math.min(0.4, (redRatio - 0.36) * 2.5);
        }

        // 2. Brightness (any light source)
        const brightness = total / 3;
        if (brightness > 10) {
            quality += Math.min(0.25, brightness / 400);
        }

        // 3. Signal oscillation (the most important indicator)
        if (state.rawSignal.length > 30) {
            const recent = state.rawSignal.slice(-60);
            const mean = recent.reduce((a, b) => a + b, 0) / recent.length;
            if (mean > 0) {
                const variance = recent.reduce((a, b) => a + (b - mean) ** 2, 0) / recent.length;
                const cv = Math.sqrt(variance) / mean;
                // Even tiny oscillations count — that's the heartbeat
                quality += Math.min(0.35, cv * 50);
            }
        }

        return Math.min(1, quality);
    }

    // ============================================
    // Filtering
    // ============================================

    /**
     * Simple DC removal using exponential moving average
     */
    function removeDC(value) {
        const alpha = 0.95;
        state.dcEstimate = alpha * state.dcEstimate + (1 - alpha) * value;
        return value - state.dcEstimate;
    }

    /**
     * 2nd-order IIR Butterworth Bandpass Filter
     * Dynamically compute coefficients based on measured FPS
     */
    function bandpassFilter(value) {
        const fs = state.measuredFPS || 30;

        // Low-pass at 4.0 Hz (240 BPM max)
        const fcLP = 4.0;
        const wLP = Math.tan(Math.PI * fcLP / fs);
        const wLP2 = wLP * wLP;
        const kLP = 1 + Math.SQRT2 * wLP + wLP2;
        const bLP = [wLP2 / kLP, 2 * wLP2 / kLP, wLP2 / kLP];
        const aLP = [1, 2 * (wLP2 - 1) / kLP, (1 - Math.SQRT2 * wLP + wLP2) / kLP];

        // Apply low-pass
        state.lpX[2] = state.lpX[1];
        state.lpX[1] = state.lpX[0];
        state.lpX[0] = value;

        const lpOut = bLP[0] * state.lpX[0] + bLP[1] * state.lpX[1] + bLP[2] * state.lpX[2]
            - aLP[1] * state.lpY[1] - aLP[2] * state.lpY[2];

        state.lpY[2] = state.lpY[1];
        state.lpY[1] = state.lpY[0];
        state.lpY[0] = isFinite(lpOut) ? lpOut : 0;

        // High-pass at 0.5 Hz (30 BPM min — wider range to catch slow hearts)
        const fcHP = 0.5;
        const wHP = Math.tan(Math.PI * fcHP / fs);
        const wHP2 = wHP * wHP;
        const kHP = 1 + Math.SQRT2 * wHP + wHP2;
        const bHP = [1 / kHP, -2 / kHP, 1 / kHP];
        const aHP = [1, 2 * (wHP2 - 1) / kHP, (1 - Math.SQRT2 * wHP + wHP2) / kHP];

        state.hpX[2] = state.hpX[1];
        state.hpX[1] = state.hpX[0];
        state.hpX[0] = state.lpY[0];

        const hpOut = bHP[0] * state.hpX[0] + bHP[1] * state.hpX[1] + bHP[2] * state.hpX[2]
            - aHP[1] * state.hpY[1] - aHP[2] * state.hpY[2];

        state.hpY[2] = state.hpY[1];
        state.hpY[1] = state.hpY[0];
        state.hpY[0] = isFinite(hpOut) ? hpOut : 0;

        return state.hpY[0];
    }

    /**
     * Moving average smoothing
     */
    function smooth(arr, windowSize) {
        if (arr.length < windowSize) return [...arr];
        const result = new Array(arr.length);
        const half = Math.floor(windowSize / 2);
        for (let i = 0; i < arr.length; i++) {
            const s = Math.max(0, i - half);
            const e = Math.min(arr.length, i + half + 1);
            let sum = 0;
            for (let j = s; j < e; j++) sum += arr[j];
            result[i] = sum / (e - s);
        }
        return result;
    }

    // ============================================
    // BPM Detection — Method 1: Peak Detection
    // ============================================
    function detectPeaks(signal) {
        if (signal.length < 6) return [];

        const peaks = [];
        const fs = state.measuredFPS || 30;
        const minDist = Math.max(3, Math.floor(fs * 60 / CONFIG.MAX_BPM));

        // Adaptive threshold: use a percentage of the max amplitude in recent data
        const recentLen = Math.min(signal.length, Math.floor(fs * 4));
        const recent = signal.slice(-recentLen);
        const maxAmp = Math.max(...recent.map(Math.abs));
        const threshold = maxAmp * 0.1; // 10% of max — very permissive

        let lastPeakIdx = -minDist;

        for (let i = 1; i < signal.length - 1; i++) {
            // Simple local maximum: greater than both neighbors
            if (
                signal[i] > signal[i - 1] &&
                signal[i] >= signal[i + 1] &&
                signal[i] > threshold &&
                (i - lastPeakIdx) >= minDist
            ) {
                peaks.push(i);
                lastPeakIdx = i;
            }
        }

        return peaks;
    }

    function bpmFromPeaks(peaks, timestamps) {
        if (peaks.length < CONFIG.MIN_PEAKS_FOR_BPM) return null;

        const intervals = [];
        for (let i = 1; i < peaks.length; i++) {
            const dt = timestamps[peaks[i]] - timestamps[peaks[i - 1]];
            if (dt > 0) {
                const bpm = 60000 / dt;
                if (bpm >= CONFIG.MIN_BPM && bpm <= CONFIG.MAX_BPM) {
                    intervals.push(dt);
                }
            }
        }

        if (intervals.length < 1) return null;

        // Median interval
        intervals.sort((a, b) => a - b);
        const median = intervals[Math.floor(intervals.length / 2)];
        state.lastIBI = Math.round(median);
        return Math.round(60000 / median);
    }

    // ============================================
    // BPM Detection — Method 2: Auto-Correlation
    // ============================================
    function bpmFromAutocorrelation(signal) {
        const fs = state.measuredFPS || 30;
        const n = signal.length;
        if (n < fs * 2) return null; // Need at least 2 seconds

        // Use last ~5 seconds of data
        const len = Math.min(n, Math.floor(fs * 5));
        const data = signal.slice(-len);

        // Normalize (zero mean, unit variance)
        const mean = data.reduce((a, b) => a + b, 0) / data.length;
        const centered = data.map(v => v - mean);
        const energy = centered.reduce((a, b) => a + b * b, 0);
        if (energy < 1e-10) return null;

        // Auto-correlation for lags corresponding to 40–220 BPM
        const minLag = Math.floor(fs * 60 / CONFIG.MAX_BPM); // ~8 samples at 30fps for 220bpm
        const maxLag = Math.floor(fs * 60 / CONFIG.MIN_BPM);  // ~45 samples at 30fps for 40bpm
        const safeLag = Math.min(maxLag, centered.length - 1);

        let bestLag = minLag;
        let bestCorr = -Infinity;

        for (let lag = minLag; lag <= safeLag; lag++) {
            let corr = 0;
            const count = centered.length - lag;
            for (let i = 0; i < count; i++) {
                corr += centered[i] * centered[i + lag];
            }
            corr /= count;

            if (corr > bestCorr) {
                bestCorr = corr;
                bestLag = lag;
            }
        }

        // The correlation should be positive and meaningful
        const normalizedCorr = bestCorr / (energy / centered.length);
        if (normalizedCorr < 0.05) return null; // Too weak

        const bpm = Math.round((fs * 60) / bestLag);
        if (bpm >= CONFIG.MIN_BPM && bpm <= CONFIG.MAX_BPM) {
            return bpm;
        }
        return null;
    }

    // ============================================
    // Main Processing Loop
    // ============================================
    function processFrame() {
        if (!state.isRunning) return;

        const now = performance.now();

        // Measure actual FPS
        state.sampleCount++;
        const totalElapsed = now - state.startTime;
        if (totalElapsed > 1000) {
            state.measuredFPS = Math.round(state.sampleCount / (totalElapsed / 1000));
        }

        // Throttle: aim for ~30 samples/sec max
        const minInterval = 1000 / 35;
        if (now - state.lastSampleTime < minInterval) {
            state.animationFrameId = requestAnimationFrame(processFrame);
            return;
        }
        state.lastSampleTime = now;

        // Extract channels
        const channels = extractChannels();
        const redValue = channels.red;

        // Signal quality
        state.signalQuality = state.signalQuality * 0.9 + assessSignalQuality(channels) * 0.1;

        // Store raw
        state.rawSignal.push(redValue);
        state.timestamps.push(now);

        // Remove DC component and apply bandpass filter
        const acComponent = removeDC(redValue);
        const filtered = bandpassFilter(acComponent);
        state.filteredSignal.push(filtered);

        // Trim buffers
        const maxSamples = Math.floor((state.measuredFPS || 30) * CONFIG.BUFFER_SECONDS);
        while (state.rawSignal.length > maxSamples) {
            state.rawSignal.shift();
            state.filteredSignal.shift();
            state.timestamps.shift();
        }

        // Process BPM after stabilization
        const timeSinceStart = now - state.startTime;

        if (timeSinceStart > CONFIG.STABILIZATION_TIME && state.filteredSignal.length > 30) {
            // Smooth the filtered signal
            const smoothed = smooth(state.filteredSignal, CONFIG.SMOOTHING_WINDOW);

            // Method 1: Peak detection
            let bpm = null;
            const peaks = detectPeaks(smoothed);
            bpm = bpmFromPeaks(peaks, state.timestamps);

            if (bpm !== null) {
                state.detectionMethod = 'peaks';
            }

            // Method 2: Auto-correlation fallback
            if (bpm === null && state.filteredSignal.length > (state.measuredFPS || 30) * 3) {
                bpm = bpmFromAutocorrelation(state.filteredSignal);
                if (bpm !== null) {
                    state.detectionMethod = 'autocorr';
                    // Estimate IBI from autocorrelation BPM
                    state.lastIBI = Math.round(60000 / bpm);
                }
            }

            if (bpm !== null) {
                // Sanity check: if we have history, reject outliers (>30% change)
                if (state.bpmHistory.length >= 3) {
                    const recentAvg = state.bpmHistory.slice(-3).reduce((a, b) => a + b, 0) / 3;
                    if (Math.abs(bpm - recentAvg) / recentAvg > 0.35) {
                        // Skip this reading — likely noise
                        bpm = null;
                    }
                }
            }

            if (bpm !== null) {
                state.bpmHistory.push(bpm);
                state.allBpmReadings.push(bpm);
                if (state.bpmHistory.length > CONFIG.BPM_HISTORY_SIZE) {
                    state.bpmHistory.shift();
                }

                // Compute smoothed BPM
                state.currentBPM = Math.round(
                    state.bpmHistory.reduce((a, b) => a + b, 0) / state.bpmHistory.length
                );

                // Track stats from all readings
                if (state.currentBPM > 0) {
                    state.minBPM = Math.min(state.minBPM, state.currentBPM);
                    state.maxBPM = Math.max(state.maxBPM, state.currentBPM);
                    // True average of all readings
                    state.avgBPM = Math.round(
                        state.allBpmReadings.reduce((a, b) => a + b, 0) / state.allBpmReadings.length
                    );
                }

                // Heartbeat animation — trigger on new peak
                if (peaks.length > 0) {
                    const lastPeakTs = state.timestamps[peaks[peaks.length - 1]];
                    if (lastPeakTs > state.lastPeakTime) {
                        state.lastPeakTime = lastPeakTs;
                        triggerHeartbeat();
                    }
                }
            }
        }

        // Update UI
        updateUI(timeSinceStart);

        // Draw waveform
        drawWaveform();

        state.animationFrameId = requestAnimationFrame(processFrame);
    }

    // ============================================
    // UI Updates
    // ============================================
    function updateUI(timeSinceStart) {
        const timeSeconds = Math.floor(timeSinceStart / 1000);
        DOM.waveformTime.textContent = `${timeSeconds}s`;

        // Signal quality display
        const qualityPct = Math.round(state.signalQuality * 100);
        DOM.sqBarFill.style.width = `${qualityPct}%`;

        const sq = DOM.sqValue;
        if (state.signalQuality > 0.55) {
            sq.textContent = 'Excellent';
            sq.className = 'sq-value good';
        } else if (state.signalQuality > 0.25) {
            sq.textContent = 'Good';
            sq.className = 'sq-value fair';
        } else if (state.signalQuality > 0.1) {
            sq.textContent = 'Weak — Hold steady';
            sq.className = 'sq-value poor';
        } else {
            sq.textContent = 'Place finger on lens';
            sq.className = 'sq-value poor';
        }

        // BPM display
        if (state.currentBPM > 0) {
            DOM.bpmValue.textContent = state.currentBPM;
            const methodLabel = state.detectionMethod === 'autocorr' ? 'Estimated' : 'Stable reading';
            DOM.bpmLabel.textContent = methodLabel;
            DOM.bpmLabel.className = 'bpm-label stable';
        } else if (timeSinceStart < CONFIG.STABILIZATION_TIME) {
            DOM.bpmValue.textContent = '--';
            DOM.bpmLabel.textContent = 'Calibrating...';
            DOM.bpmLabel.className = 'bpm-label';
        } else if (state.signalQuality < 0.1) {
            DOM.bpmValue.textContent = '--';
            DOM.bpmLabel.textContent = 'Cover the camera with your finger';
            DOM.bpmLabel.className = 'bpm-label';
        } else {
            DOM.bpmValue.textContent = '--';
            DOM.bpmLabel.textContent = 'Analyzing signal...';
            DOM.bpmLabel.className = 'bpm-label';
        }

        // Stats
        DOM.statAvg.textContent = state.avgBPM > 0 ? state.avgBPM : '--';
        DOM.statMin.textContent = state.minBPM < Infinity ? state.minBPM : '--';
        DOM.statMax.textContent = state.maxBPM > 0 ? state.maxBPM : '--';
        DOM.statIBI.textContent = state.lastIBI > 0 ? state.lastIBI : '--';
    }

    function triggerHeartbeat() {
        DOM.bpmHeart.classList.remove('beat');
        void DOM.bpmHeart.offsetWidth;
        DOM.bpmHeart.classList.add('beat');
    }

    // ============================================
    // Waveform Rendering
    // ============================================
    function drawWaveform() {
        const canvas = DOM.waveformCanvas;
        const dpr = window.devicePixelRatio || 1;
        const rect = canvas.getBoundingClientRect();

        if (canvas.width !== Math.round(rect.width * dpr) || canvas.height !== Math.round(rect.height * dpr)) {
            canvas.width = Math.round(rect.width * dpr);
            canvas.height = Math.round(rect.height * dpr);
            waveCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
        }

        const w = rect.width;
        const h = rect.height;

        waveCtx.clearRect(0, 0, w, h);

        // Grid
        waveCtx.strokeStyle = 'rgba(255, 255, 255, 0.04)';
        waveCtx.lineWidth = 1;
        for (let i = 1; i < 4; i++) {
            waveCtx.beginPath();
            waveCtx.moveTo(0, (h / 4) * i);
            waveCtx.lineTo(w, (h / 4) * i);
            waveCtx.stroke();
        }

        // Center line
        waveCtx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
        waveCtx.beginPath();
        waveCtx.moveTo(0, h / 2);
        waveCtx.lineTo(w, h / 2);
        waveCtx.stroke();

        const signal = state.filteredSignal;
        if (signal.length < 3) return;

        // Show last ~5 seconds
        const fs = state.measuredFPS || 30;
        const displaySamples = Math.min(signal.length, Math.floor(fs * 5));
        const displayData = signal.slice(-displaySamples);

        // Auto-scale with padding
        let maxVal = -Infinity, minVal = Infinity;
        for (const v of displayData) {
            if (v > maxVal) maxVal = v;
            if (v < minVal) minVal = v;
        }
        const range = (maxVal - minVal) || 1;
        const pad = 0.12;
        const yScale = h * (1 - 2 * pad) / range;
        const yOffset = h * pad - minVal * yScale;

        // Helper to map data point to canvas Y
        const toY = (val) => h - (val * yScale + yOffset);
        const toX = (i) => (i / (displaySamples - 1)) * w;

        // --- Filled area ---
        const fillGrad = waveCtx.createLinearGradient(0, 0, 0, h);
        fillGrad.addColorStop(0, 'rgba(0, 212, 216, 0.18)');
        fillGrad.addColorStop(0.5, 'rgba(0, 212, 216, 0.05)');
        fillGrad.addColorStop(1, 'rgba(0, 212, 216, 0.0)');

        waveCtx.beginPath();
        waveCtx.moveTo(toX(0), h);
        waveCtx.lineTo(toX(0), toY(displayData[0]));
        for (let i = 1; i < displayData.length; i++) {
            const cx = (toX(i - 1) + toX(i)) / 2;
            const cy = (toY(displayData[i - 1]) + toY(displayData[i])) / 2;
            waveCtx.quadraticCurveTo(toX(i - 1), toY(displayData[i - 1]), cx, cy);
        }
        waveCtx.lineTo(toX(displayData.length - 1), toY(displayData[displayData.length - 1]));
        waveCtx.lineTo(toX(displayData.length - 1), h);
        waveCtx.closePath();
        waveCtx.fillStyle = fillGrad;
        waveCtx.fill();

        // --- Waveform stroke ---
        waveCtx.beginPath();
        waveCtx.moveTo(toX(0), toY(displayData[0]));
        for (let i = 1; i < displayData.length; i++) {
            const cx = (toX(i - 1) + toX(i)) / 2;
            const cy = (toY(displayData[i - 1]) + toY(displayData[i])) / 2;
            waveCtx.quadraticCurveTo(toX(i - 1), toY(displayData[i - 1]), cx, cy);
        }

        const strokeGrad = waveCtx.createLinearGradient(0, 0, w, 0);
        strokeGrad.addColorStop(0, 'rgba(0, 212, 216, 0.25)');
        strokeGrad.addColorStop(0.6, 'rgba(0, 212, 216, 0.85)');
        strokeGrad.addColorStop(1, 'rgba(0, 212, 216, 1)');

        waveCtx.strokeStyle = strokeGrad;
        waveCtx.lineWidth = 2.2;
        waveCtx.lineJoin = 'round';
        waveCtx.lineCap = 'round';
        waveCtx.stroke();

        // --- Glow dot at end ---
        const endX = toX(displayData.length - 1);
        const endY = toY(displayData[displayData.length - 1]);

        const glowGrad = waveCtx.createRadialGradient(endX, endY, 0, endX, endY, 10);
        glowGrad.addColorStop(0, 'rgba(0, 212, 216, 0.7)');
        glowGrad.addColorStop(1, 'rgba(0, 212, 216, 0)');
        waveCtx.beginPath();
        waveCtx.arc(endX, endY, 10, 0, Math.PI * 2);
        waveCtx.fillStyle = glowGrad;
        waveCtx.fill();

        waveCtx.beginPath();
        waveCtx.arc(endX, endY, 3, 0, Math.PI * 2);
        waveCtx.fillStyle = '#00d4d8';
        waveCtx.fill();
    }

    // ============================================
    // Start / Stop
    // ============================================
    async function startMeasurement() {
        DOM.btnStart.disabled = true;
        DOM.btnStart.textContent = 'Starting...';

        const success = await startCamera();
        if (!success) {
            DOM.btnStart.disabled = false;
            DOM.btnStart.textContent = 'Start Measurement';
            return;
        }

        // Reset all state
        state.isRunning = true;
        state.startTime = performance.now();
        state.lastSampleTime = 0;
        state.sampleCount = 0;
        state.measuredFPS = 30;
        state.rawSignal = [];
        state.filteredSignal = [];
        state.timestamps = [];
        state.bpmHistory = [];
        state.allBpmReadings = [];
        state.currentBPM = 0;
        state.avgBPM = 0;
        state.minBPM = Infinity;
        state.maxBPM = 0;
        state.lastIBI = 0;
        state.signalQuality = 0;
        state.lastPeakTime = 0;
        state.detectionMethod = '';
        state.dcEstimate = 0;
        state.lpX = [0, 0, 0];
        state.lpY = [0, 0, 0];
        state.hpX = [0, 0, 0];
        state.hpY = [0, 0, 0];

        // Swap views
        DOM.instructionCard.style.display = 'none';
        DOM.measurementPanel.style.display = 'flex';

        // Reset displays
        DOM.bpmValue.textContent = '--';
        DOM.bpmLabel.textContent = 'Calibrating...';
        DOM.bpmLabel.className = 'bpm-label';
        DOM.sqBarFill.style.width = '0%';
        DOM.sqValue.textContent = 'Waiting...';
        DOM.sqValue.className = 'sq-value';

        DOM.btnStart.disabled = false;
        DOM.btnStart.innerHTML = `
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="20" height="20">
                <circle cx="12" cy="12" r="10"/>
                <polygon points="10,8 16,12 10,16" fill="currentColor" stroke="none"/>
            </svg>
            Start Measurement`;

        state.animationFrameId = requestAnimationFrame(processFrame);
    }

    function stopMeasurement() {
        state.isRunning = false;

        if (state.animationFrameId) {
            cancelAnimationFrame(state.animationFrameId);
            state.animationFrameId = null;
        }

        stopCamera();

        DOM.measurementPanel.style.display = 'none';
        DOM.instructionCard.style.display = '';

        DOM.instructionCard.style.animation = 'none';
        void DOM.instructionCard.offsetWidth;
        DOM.instructionCard.style.animation = '';
    }

    // ============================================
    // Event Listeners
    // ============================================
    DOM.btnStart.addEventListener('click', startMeasurement);
    DOM.btnStop.addEventListener('click', stopMeasurement);

    document.addEventListener('visibilitychange', () => {
        if (document.hidden && state.isRunning) {
            stopMeasurement();
        }
    });

    window.addEventListener('resize', () => {
        if (state.isRunning) drawWaveform();
    });

    // Log for debugging
    console.log('PulseSense initialized. Click "Start Measurement" to begin.');

})();
