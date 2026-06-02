/**
 * PulseSense — Camera-based Heart Rate Monitor
 * 
 * Uses Photoplethysmography (PPG) to detect heart rate from the camera.
 * The red channel intensity fluctuates with each heartbeat as blood volume
 * changes in the fingertip tissue.
 */

(function () {
    'use strict';

    // ============================================
    // Configuration
    // ============================================
    const CONFIG = {
        SAMPLE_RATE: 30,            // Target FPS for sampling
        BUFFER_SIZE: 300,           // ~10 seconds of data at 30fps
        MIN_BPM: 40,
        MAX_BPM: 210,
        STABILIZATION_TIME: 3000,   // ms before showing BPM
        MIN_PEAKS_FOR_BPM: 3,      // Minimum peaks needed to calculate BPM
        SMOOTHING_WINDOW: 5,        // Moving average window
        QUALITY_THRESHOLD: 0.3,     // Minimum signal quality to show BPM
        PEAK_PROMINENCE: 0.15,      // Minimum prominence for peak detection
        BPM_HISTORY_SIZE: 8,        // Number of BPM readings to average
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

        // Signal buffers
        rawSignal: [],          // Raw red channel averages
        filteredSignal: [],     // After bandpass filtering
        timestamps: [],         // Timestamps for each sample

        // Results
        bpmHistory: [],
        currentBPM: 0,
        avgBPM: 0,
        minBPM: Infinity,
        maxBPM: 0,
        lastIBI: 0,
        signalQuality: 0,
        lastPeakTime: 0,

        // Filter state (IIR Butterworth)
        filterState: {
            x: [0, 0, 0],      // Input history
            y: [0, 0, 0],      // Output history
            xHigh: [0, 0, 0],
            yHigh: [0, 0, 0],
        }
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
            // Request camera with torch (flash)
            const constraints = {
                video: {
                    facingMode: 'environment',
                    width: { ideal: 320 },
                    height: { ideal: 240 },
                }
            };

            state.stream = await navigator.mediaDevices.getUserMedia(constraints);
            DOM.video.srcObject = state.stream;

            // Wait for video to be ready
            await new Promise((resolve) => {
                DOM.video.onloadedmetadata = () => {
                    DOM.video.play();
                    resolve();
                };
            });

            // Try to enable torch/flash
            const track = state.stream.getVideoTracks()[0];
            const capabilities = track.getCapabilities ? track.getCapabilities() : {};

            if (capabilities.torch) {
                await track.applyConstraints({
                    advanced: [{ torch: true }]
                });
            }

            // Set canvas size to match video
            DOM.canvas.width = DOM.video.videoWidth;
            DOM.canvas.height = DOM.video.videoHeight;

            return true;
        } catch (err) {
            console.error('Camera access error:', err);
            alert('Unable to access camera. Please ensure camera permissions are granted and try again.');
            return false;
        }
    }

    function stopCamera() {
        if (state.stream) {
            state.stream.getTracks().forEach(track => track.stop());
            state.stream = null;
        }
        DOM.video.srcObject = null;
    }

    // ============================================
    // Signal Processing
    // ============================================

    /**
     * Extract the average red channel value from the current video frame.
     * When a finger covers the camera with flash on, the red channel
     * dominates and fluctuates with blood volume changes.
     */
    function extractRedChannel() {
        ctx.drawImage(DOM.video, 0, 0, DOM.canvas.width, DOM.canvas.height);
        const imageData = ctx.getImageData(0, 0, DOM.canvas.width, DOM.canvas.height);
        const pixels = imageData.data;
        const len = pixels.length;

        let redSum = 0;
        let greenSum = 0;
        let blueSum = 0;
        let count = 0;

        // Sample center region (more reliable, less noise from edges)
        const w = DOM.canvas.width;
        const h = DOM.canvas.height;
        const margin = 0.2; // 20% margin on each side
        const x0 = Math.floor(w * margin);
        const x1 = Math.floor(w * (1 - margin));
        const y0 = Math.floor(h * margin);
        const y1 = Math.floor(h * (1 - margin));

        for (let y = y0; y < y1; y++) {
            for (let x = x0; x < x1; x++) {
                const i = (y * w + x) * 4;
                redSum += pixels[i];
                greenSum += pixels[i + 1];
                blueSum += pixels[i + 2];
                count++;
            }
        }

        return {
            red: redSum / count,
            green: greenSum / count,
            blue: blueSum / count,
        };
    }

    /**
     * Assess signal quality based on:
     * - Red channel dominance (finger on camera makes red dominant)
     * - Signal variance (good PPG has clear oscillations)
     */
    function assessSignalQuality(channels) {
        const { red, green, blue } = channels;
        const total = red + green + blue;
        if (total < 1) return 0;

        const redRatio = red / total;
        const brightness = total / 3;

        // Finger on camera: red > 60% of total, brightness > 50
        let quality = 0;

        // Red dominance score (0–0.5)
        if (redRatio > 0.45) {
            quality += Math.min(0.5, (redRatio - 0.45) * 3);
        }

        // Brightness score (0–0.3)
        if (brightness > 30) {
            quality += Math.min(0.3, (brightness - 30) / 200);
        }

        // Signal variance score (0–0.2) — needs some samples
        if (state.rawSignal.length > 20) {
            const recent = state.rawSignal.slice(-30);
            const mean = recent.reduce((a, b) => a + b, 0) / recent.length;
            const variance = recent.reduce((a, b) => a + (b - mean) ** 2, 0) / recent.length;
            const cv = Math.sqrt(variance) / (mean || 1); // coefficient of variation
            if (cv > 0.001) {
                quality += Math.min(0.2, cv * 20);
            }
        }

        return Math.min(1, quality);
    }

    /**
     * 2nd-order IIR Bandpass filter
     * Passband: ~0.7 Hz (42 BPM) to ~3.5 Hz (210 BPM)
     * Designed for 30 Hz sampling rate
     */
    function bandpassFilter(value) {
        // Low-pass filter at ~3.5 Hz (Butterworth 2nd order, fs=30)
        // Coefficients computed for cutoff = 3.5/15 (normalized)
        const aLP = [1, -1.1430, 0.4128];
        const bLP = [0.0675, 0.1349, 0.0675];

        // Shift input history
        state.filterState.x[2] = state.filterState.x[1];
        state.filterState.x[1] = state.filterState.x[0];
        state.filterState.x[0] = value;

        // Low-pass
        let lpOut =
            bLP[0] * state.filterState.x[0] +
            bLP[1] * state.filterState.x[1] +
            bLP[2] * state.filterState.x[2] -
            aLP[1] * state.filterState.y[1] -
            aLP[2] * state.filterState.y[2];

        state.filterState.y[2] = state.filterState.y[1];
        state.filterState.y[1] = state.filterState.y[0];
        state.filterState.y[0] = lpOut;

        // High-pass filter at ~0.7 Hz (Butterworth 2nd order, fs=30)
        const aHP = [1, -1.8227, 0.8372];
        const bHP = [0.9152, -1.8305, 0.9152];

        state.filterState.xHigh[2] = state.filterState.xHigh[1];
        state.filterState.xHigh[1] = state.filterState.xHigh[0];
        state.filterState.xHigh[0] = lpOut;

        let hpOut =
            bHP[0] * state.filterState.xHigh[0] +
            bHP[1] * state.filterState.xHigh[1] +
            bHP[2] * state.filterState.xHigh[2] -
            aHP[1] * state.filterState.yHigh[1] -
            aHP[2] * state.filterState.yHigh[2];

        state.filterState.yHigh[2] = state.filterState.yHigh[1];
        state.filterState.yHigh[1] = state.filterState.yHigh[0];
        state.filterState.yHigh[0] = hpOut;

        return hpOut;
    }

    /**
     * Simple moving average smoother
     */
    function movingAverage(arr, windowSize) {
        if (arr.length < windowSize) return arr;
        const result = [];
        for (let i = 0; i < arr.length; i++) {
            const start = Math.max(0, i - Math.floor(windowSize / 2));
            const end = Math.min(arr.length, i + Math.floor(windowSize / 2) + 1);
            const window = arr.slice(start, end);
            result.push(window.reduce((a, b) => a + b, 0) / window.length);
        }
        return result;
    }

    /**
     * Detect peaks in the filtered signal using adaptive thresholding
     */
    function detectPeaks(signal) {
        if (signal.length < 10) return [];

        const peaks = [];
        const minDistance = Math.floor(CONFIG.SAMPLE_RATE * 60 / CONFIG.MAX_BPM); // min samples between peaks
        const maxDistance = Math.floor(CONFIG.SAMPLE_RATE * 60 / CONFIG.MIN_BPM);

        // Compute adaptive threshold
        const absSignal = signal.map(Math.abs);
        const maxAmp = Math.max(...absSignal.slice(-CONFIG.SAMPLE_RATE * 3)); // last 3 seconds
        const threshold = maxAmp * CONFIG.PEAK_PROMINENCE;

        let lastPeakIdx = -minDistance;

        for (let i = 2; i < signal.length - 2; i++) {
            // Local maximum check
            if (
                signal[i] > signal[i - 1] &&
                signal[i] > signal[i - 2] &&
                signal[i] > signal[i + 1] &&
                signal[i] > signal[i + 2] &&
                signal[i] > threshold &&
                (i - lastPeakIdx) >= minDistance
            ) {
                peaks.push(i);
                lastPeakIdx = i;
            }
        }

        return peaks;
    }

    /**
     * Calculate BPM from peak intervals
     */
    function calculateBPM(peaks, timestamps) {
        if (peaks.length < CONFIG.MIN_PEAKS_FOR_BPM) return null;

        // Get intervals between consecutive peaks
        const intervals = [];
        for (let i = 1; i < peaks.length; i++) {
            const dt = timestamps[peaks[i]] - timestamps[peaks[i - 1]];
            if (dt > 0) {
                const instantBPM = 60000 / dt; // ms to BPM
                if (instantBPM >= CONFIG.MIN_BPM && instantBPM <= CONFIG.MAX_BPM) {
                    intervals.push(dt);
                }
            }
        }

        if (intervals.length < 2) return null;

        // Use median for robustness against outliers
        intervals.sort((a, b) => a - b);
        const median = intervals[Math.floor(intervals.length / 2)];
        const bpm = Math.round(60000 / median);

        // Store the last inter-beat interval
        state.lastIBI = Math.round(median);

        return bpm;
    }

    // ============================================
    // Main Processing Loop
    // ============================================
    function processFrame(timestamp) {
        if (!state.isRunning) return;

        const now = performance.now();
        const elapsed = now - state.lastSampleTime;
        const targetInterval = 1000 / CONFIG.SAMPLE_RATE;

        // Throttle to target sample rate
        if (elapsed >= targetInterval * 0.8) {
            state.lastSampleTime = now;

            // Extract red channel
            const channels = extractRedChannel();
            const redValue = channels.red;

            // Assess signal quality
            state.signalQuality = assessSignalQuality(channels);

            // Store raw value
            state.rawSignal.push(redValue);
            state.timestamps.push(now);

            // Apply bandpass filter
            const filtered = bandpassFilter(redValue);
            state.filteredSignal.push(filtered);

            // Trim buffers to max size
            if (state.rawSignal.length > CONFIG.BUFFER_SIZE) {
                state.rawSignal.shift();
                state.filteredSignal.shift();
                state.timestamps.shift();
            }

            // Process after stabilization period
            const timeSinceStart = now - state.startTime;

            if (timeSinceStart > CONFIG.STABILIZATION_TIME && state.signalQuality > CONFIG.QUALITY_THRESHOLD) {
                // Smooth filtered signal
                const smoothed = movingAverage(state.filteredSignal, CONFIG.SMOOTHING_WINDOW);

                // Detect peaks
                const peaks = detectPeaks(smoothed);

                // Calculate BPM
                const bpm = calculateBPM(peaks, state.timestamps);

                if (bpm !== null) {
                    // Add to history and compute running average
                    state.bpmHistory.push(bpm);
                    if (state.bpmHistory.length > CONFIG.BPM_HISTORY_SIZE) {
                        state.bpmHistory.shift();
                    }

                    // Average the history for stability
                    state.currentBPM = Math.round(
                        state.bpmHistory.reduce((a, b) => a + b, 0) / state.bpmHistory.length
                    );

                    // Update min/max/avg
                    if (state.currentBPM > 0) {
                        state.minBPM = Math.min(state.minBPM, state.currentBPM);
                        state.maxBPM = Math.max(state.maxBPM, state.currentBPM);
                        state.avgBPM = state.currentBPM; // running avg from history
                    }

                    // Trigger heartbeat animation on new peak
                    if (peaks.length > 0) {
                        const lastPeakTimestamp = state.timestamps[peaks[peaks.length - 1]];
                        if (lastPeakTimestamp > state.lastPeakTime) {
                            state.lastPeakTime = lastPeakTimestamp;
                            triggerHeartbeat();
                        }
                    }
                }
            }

            // Update UI
            updateUI(timeSinceStart);
        }

        // Draw waveform every frame for smoothness
        drawWaveform();

        state.animationFrameId = requestAnimationFrame(processFrame);
    }

    // ============================================
    // UI Updates
    // ============================================
    function updateUI(timeSinceStart) {
        const timeSeconds = Math.floor(timeSinceStart / 1000);
        DOM.waveformTime.textContent = `${timeSeconds}s`;

        // Signal quality
        const qualityPct = Math.round(state.signalQuality * 100);
        DOM.sqBarFill.style.width = `${qualityPct}%`;

        const sqValueEl = DOM.sqValue;
        if (state.signalQuality > 0.6) {
            sqValueEl.textContent = 'Excellent';
            sqValueEl.className = 'sq-value good';
        } else if (state.signalQuality > 0.35) {
            sqValueEl.textContent = 'Good';
            sqValueEl.className = 'sq-value fair';
        } else if (state.signalQuality > 0.15) {
            sqValueEl.textContent = 'Weak — Press harder';
            sqValueEl.className = 'sq-value poor';
        } else {
            sqValueEl.textContent = 'No finger detected';
            sqValueEl.className = 'sq-value poor';
        }

        // BPM
        if (state.currentBPM > 0 && state.signalQuality > CONFIG.QUALITY_THRESHOLD) {
            DOM.bpmValue.textContent = state.currentBPM;
            DOM.bpmLabel.textContent = 'Stable reading';
            DOM.bpmLabel.className = 'bpm-label stable';
        } else if (timeSinceStart > CONFIG.STABILIZATION_TIME) {
            DOM.bpmValue.textContent = '--';
            DOM.bpmLabel.textContent = 'Place finger on camera';
            DOM.bpmLabel.className = 'bpm-label';
        } else {
            DOM.bpmValue.textContent = '--';
            DOM.bpmLabel.textContent = 'Calibrating...';
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
        // Force reflow to restart animation
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

        // Set canvas resolution
        if (canvas.width !== rect.width * dpr || canvas.height !== rect.height * dpr) {
            canvas.width = rect.width * dpr;
            canvas.height = rect.height * dpr;
            waveCtx.scale(dpr, dpr);
        }

        const w = rect.width;
        const h = rect.height;

        // Clear
        waveCtx.clearRect(0, 0, w, h);

        // Draw grid lines
        waveCtx.strokeStyle = 'rgba(255, 255, 255, 0.04)';
        waveCtx.lineWidth = 1;
        for (let i = 1; i < 4; i++) {
            const y = (h / 4) * i;
            waveCtx.beginPath();
            waveCtx.moveTo(0, y);
            waveCtx.lineTo(w, y);
            waveCtx.stroke();
        }

        // Draw center line
        waveCtx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
        waveCtx.beginPath();
        waveCtx.moveTo(0, h / 2);
        waveCtx.lineTo(w, h / 2);
        waveCtx.stroke();

        const signal = state.filteredSignal;
        if (signal.length < 2) return;

        // Determine display range (show last ~5 seconds)
        const displaySamples = Math.min(signal.length, CONFIG.SAMPLE_RATE * 5);
        const startIdx = signal.length - displaySamples;
        const displayData = signal.slice(startIdx);

        // Auto-scale
        let maxVal = -Infinity;
        let minVal = Infinity;
        for (let i = 0; i < displayData.length; i++) {
            if (displayData[i] > maxVal) maxVal = displayData[i];
            if (displayData[i] < minVal) minVal = displayData[i];
        }

        const range = maxVal - minVal || 1;
        const padding = 0.15;
        const yScale = h * (1 - 2 * padding) / range;
        const yOffset = h * padding - minVal * yScale;

        // Draw filled area
        const gradient = waveCtx.createLinearGradient(0, 0, 0, h);
        gradient.addColorStop(0, 'rgba(0, 212, 216, 0.20)');
        gradient.addColorStop(0.5, 'rgba(0, 212, 216, 0.06)');
        gradient.addColorStop(1, 'rgba(0, 212, 216, 0.0)');

        waveCtx.beginPath();
        waveCtx.moveTo(0, h);

        for (let i = 0; i < displayData.length; i++) {
            const x = (i / (displaySamples - 1)) * w;
            const y = h - (displayData[i] * yScale + yOffset);
            if (i === 0) {
                waveCtx.lineTo(x, y);
            } else {
                // Smooth curve using bezier
                const prevX = ((i - 1) / (displaySamples - 1)) * w;
                const prevY = h - (displayData[i - 1] * yScale + yOffset);
                const cpX = (prevX + x) / 2;
                waveCtx.quadraticCurveTo(prevX, prevY, cpX, (prevY + y) / 2);
            }
        }

        // Last point
        const lastX = w;
        const lastY = h - (displayData[displayData.length - 1] * yScale + yOffset);
        waveCtx.lineTo(lastX, lastY);
        waveCtx.lineTo(w, h);
        waveCtx.closePath();
        waveCtx.fillStyle = gradient;
        waveCtx.fill();

        // Draw waveform line
        waveCtx.beginPath();
        for (let i = 0; i < displayData.length; i++) {
            const x = (i / (displaySamples - 1)) * w;
            const y = h - (displayData[i] * yScale + yOffset);
            if (i === 0) {
                waveCtx.moveTo(x, y);
            } else {
                const prevX = ((i - 1) / (displaySamples - 1)) * w;
                const prevY = h - (displayData[i - 1] * yScale + yOffset);
                const cpX = (prevX + x) / 2;
                waveCtx.quadraticCurveTo(prevX, prevY, cpX, (prevY + y) / 2);
            }
        }

        // Gradient stroke
        const strokeGrad = waveCtx.createLinearGradient(0, 0, w, 0);
        strokeGrad.addColorStop(0, 'rgba(0, 212, 216, 0.3)');
        strokeGrad.addColorStop(0.5, 'rgba(0, 212, 216, 0.9)');
        strokeGrad.addColorStop(1, 'rgba(0, 212, 216, 1)');

        waveCtx.strokeStyle = strokeGrad;
        waveCtx.lineWidth = 2;
        waveCtx.lineJoin = 'round';
        waveCtx.lineCap = 'round';
        waveCtx.stroke();

        // Draw glow on the latest point
        const glowX = w;
        const glowY = h - (displayData[displayData.length - 1] * yScale + yOffset);

        const glowGrad = waveCtx.createRadialGradient(glowX, glowY, 0, glowX, glowY, 8);
        glowGrad.addColorStop(0, 'rgba(0, 212, 216, 0.8)');
        glowGrad.addColorStop(1, 'rgba(0, 212, 216, 0)');

        waveCtx.beginPath();
        waveCtx.arc(glowX, glowY, 8, 0, Math.PI * 2);
        waveCtx.fillStyle = glowGrad;
        waveCtx.fill();

        // Dot at the end
        waveCtx.beginPath();
        waveCtx.arc(glowX, glowY, 3, 0, Math.PI * 2);
        waveCtx.fillStyle = '#00d4d8';
        waveCtx.fill();
    }

    // ============================================
    // Start / Stop
    // ============================================
    async function startMeasurement() {
        const success = await startCamera();
        if (!success) return;

        // Reset state
        state.isRunning = true;
        state.startTime = performance.now();
        state.lastSampleTime = performance.now();
        state.rawSignal = [];
        state.filteredSignal = [];
        state.timestamps = [];
        state.bpmHistory = [];
        state.currentBPM = 0;
        state.avgBPM = 0;
        state.minBPM = Infinity;
        state.maxBPM = 0;
        state.lastIBI = 0;
        state.signalQuality = 0;
        state.lastPeakTime = 0;
        state.filterState = {
            x: [0, 0, 0],
            y: [0, 0, 0],
            xHigh: [0, 0, 0],
            yHigh: [0, 0, 0],
        };

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

        // Start processing loop
        state.animationFrameId = requestAnimationFrame(processFrame);
    }

    function stopMeasurement() {
        state.isRunning = false;

        if (state.animationFrameId) {
            cancelAnimationFrame(state.animationFrameId);
            state.animationFrameId = null;
        }

        stopCamera();

        // Swap views
        DOM.measurementPanel.style.display = 'none';
        DOM.instructionCard.style.display = '';

        // Re-animate entry
        DOM.instructionCard.style.animation = 'none';
        void DOM.instructionCard.offsetWidth;
        DOM.instructionCard.style.animation = '';
    }

    // ============================================
    // Event Listeners
    // ============================================
    DOM.btnStart.addEventListener('click', startMeasurement);
    DOM.btnStop.addEventListener('click', stopMeasurement);

    // Handle page visibility (stop camera when tab is hidden)
    document.addEventListener('visibilitychange', () => {
        if (document.hidden && state.isRunning) {
            stopMeasurement();
        }
    });

    // Resize waveform canvas on window resize
    window.addEventListener('resize', () => {
        if (state.isRunning) {
            drawWaveform();
        }
    });

})();
