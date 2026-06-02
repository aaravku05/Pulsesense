/**
 * PulseSense — Camera-based Heart Rate Monitor
 * 
 * Uses Photoplethysmography (PPG) to detect heart rate.
 * 
 * Approach: Simple & robust signal processing
 * 1. Extract average red channel from camera frames
 * 2. Detrend using slow moving average subtraction (no IIR filters)
 * 3. Smooth with fast moving average
 * 4. Find peaks with adaptive thresholding
 * 5. Validate periodicity before showing BPM
 */

(function () {
    'use strict';

    // ============================================
    // Configuration
    // ============================================
    const CONFIG = {
        BUFFER_SECONDS: 12,
        MIN_BPM: 40,
        MAX_BPM: 200,
        WARMUP_SAMPLES: 60,         // ~2 seconds of data before trying BPM
        MIN_PEAKS_FOR_BPM: 3,
        BPM_HISTORY_SIZE: 5,
        DETREND_WINDOW_SEC: 3,      // Slow moving avg window for DC removal
        SMOOTH_WINDOW: 5,           // Fast moving avg for noise reduction
        IBI_CONSISTENCY: 0.40,      // Max allowed coefficient of variation in intervals
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

        // Raw data
        rawRed: [],                 // Raw red channel averages
        timestamps: [],             // performance.now() timestamps

        // Processed
        detrendedSignal: [],        // After DC removal + smoothing

        // Results
        bpmHistory: [],
        allBpmReadings: [],
        currentBPM: 0,
        avgBPM: 0,
        minBPM: Infinity,
        maxBPM: 0,
        lastIBI: 0,
        signalQuality: 0,
        lastBeatTime: 0,
        fingerDetected: false,

        // Debug
        debugRedMean: 0,
        debugRedRatio: 0,
        debugPeakCount: 0,
        debugAmplitude: 0,
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
            let stream = null;

            // Try rear camera with flash first
            try {
                stream = await navigator.mediaDevices.getUserMedia({
                    video: {
                        facingMode: { ideal: 'environment' },
                        width: { ideal: 160 },   // Small res = faster pixel reading
                        height: { ideal: 120 },
                    }
                });
            } catch (e) {
                // Fallback: any camera
                stream = await navigator.mediaDevices.getUserMedia({
                    video: { width: { ideal: 160 }, height: { ideal: 120 } }
                });
            }

            state.stream = stream;
            DOM.video.srcObject = stream;

            await new Promise(resolve => {
                DOM.video.onloadedmetadata = () => { DOM.video.play(); resolve(); };
            });

            // Enable torch if available
            const track = stream.getVideoTracks()[0];
            try {
                const caps = track.getCapabilities ? track.getCapabilities() : {};
                if (caps.torch) {
                    await track.applyConstraints({ advanced: [{ torch: true }] });
                    console.log('[PulseSense] Torch ON');
                }
            } catch (e) {
                console.log('[PulseSense] No torch available');
            }

            DOM.canvas.width = DOM.video.videoWidth || 160;
            DOM.canvas.height = DOM.video.videoHeight || 120;

            console.log(`[PulseSense] Camera: ${DOM.canvas.width}x${DOM.canvas.height}`);
            return true;
        } catch (err) {
            console.error('[PulseSense] Camera error:', err);
            alert('Camera access denied. Please allow camera permissions and try again.');
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
    function sampleRedChannel() {
        const w = DOM.canvas.width;
        const h = DOM.canvas.height;
        if (!w || !h) return null;

        ctx.drawImage(DOM.video, 0, 0, w, h);
        const data = ctx.getImageData(0, 0, w, h).data;

        let rSum = 0, gSum = 0, bSum = 0, count = 0;

        // Sample center 50% region for best signal
        const x0 = Math.floor(w * 0.25), x1 = Math.floor(w * 0.75);
        const y0 = Math.floor(h * 0.25), y1 = Math.floor(h * 0.75);

        for (let y = y0; y < y1; y++) {
            for (let x = x0; x < x1; x++) {
                const i = (y * w + x) * 4;
                rSum += data[i];
                gSum += data[i + 1];
                bSum += data[i + 2];
                count++;
            }
        }

        if (count === 0) return null;

        const r = rSum / count;
        const g = gSum / count;
        const b = bSum / count;
        const total = r + g + b;

        state.debugRedMean = r;
        state.debugRedRatio = total > 0 ? r / total : 0;

        return { red: r, green: g, blue: b };
    }

    // ============================================
    // Finger Detection
    // ============================================
    function isFingerOnCamera(channels) {
        if (!channels) return false;
        const { red, green, blue } = channels;
        const total = red + green + blue;
        if (total < 10) return false;

        const redRatio = red / total;
        const brightness = total / 3;

        // Finger on camera with flash:
        //   - Red strongly dominates (ratio > 0.4)
        //   - Brightness is high (> 50)
        // Finger on camera without flash (laptop):
        //   - Brightness drops significantly (< 30) because finger blocks light
        //   - Or red still dominates somewhat

        // Method 1: Red dominance (phone with flash)
        if (redRatio > 0.4 && brightness > 40) return true;

        // Method 2: Very red image (finger lit by flash)
        if (red > 100 && redRatio > 0.38) return true;

        // Method 3: Dark image (finger blocking laptop camera)
        // Check if brightness dropped significantly from initial samples
        if (state.rawRed.length > 10) {
            const initialBrightness = state.rawRed.slice(0, 5).reduce((a, b) => a + b, 0) / 5;
            if (brightness < initialBrightness * 0.5 && brightness < 80) return true;
        }

        return false;
    }

    // ============================================
    // Signal Processing (Simple & Robust)
    // ============================================

    /**
     * Detrend signal by subtracting a slow moving average.
     * This removes DC offset and slow drift, leaving only the AC (pulsatile) component.
     */
    function detrend(signal, windowSize) {
        const result = new Float64Array(signal.length);
        for (let i = 0; i < signal.length; i++) {
            const halfWin = Math.floor(windowSize / 2);
            const start = Math.max(0, i - halfWin);
            const end = Math.min(signal.length, i + halfWin + 1);
            let sum = 0;
            for (let j = start; j < end; j++) sum += signal[j];
            result[i] = signal[i] - sum / (end - start);
        }
        return result;
    }

    /**
     * Simple moving average smoothing
     */
    function smooth(signal, windowSize) {
        const result = new Float64Array(signal.length);
        const half = Math.floor(windowSize / 2);
        for (let i = 0; i < signal.length; i++) {
            const s = Math.max(0, i - half);
            const e = Math.min(signal.length, i + half + 1);
            let sum = 0;
            for (let j = s; j < e; j++) sum += signal[j];
            result[i] = sum / (e - s);
        }
        return result;
    }

    /**
     * Process the raw signal:
     * 1. Detrend (remove DC with 3-second moving avg)
     * 2. Smooth (reduce high-frequency noise)
     */
    function processSignal() {
        const fps = getEffectiveFPS();
        const detrendWindow = Math.max(10, Math.floor(fps * CONFIG.DETREND_WINDOW_SEC));

        // Step 1: Detrend
        const detrended = detrend(state.rawRed, detrendWindow);

        // Step 2: Smooth
        const smoothed = smooth(detrended, CONFIG.SMOOTH_WINDOW);

        // Convert to regular array for storage
        state.detrendedSignal = Array.from(smoothed);

        return smoothed;
    }

    // ============================================
    // Peak Detection
    // ============================================
    function findPeaks(signal) {
        if (signal.length < 10) return [];

        const fps = getEffectiveFPS();
        const minPeakDist = Math.max(4, Math.floor(fps * 60 / CONFIG.MAX_BPM));

        // Compute signal amplitude for thresholding
        const len = signal.length;
        const recent = Math.min(len, Math.floor(fps * 4));
        let maxAmp = 0;
        for (let i = len - recent; i < len; i++) {
            const abs = Math.abs(signal[i]);
            if (abs > maxAmp) maxAmp = abs;
        }

        // Threshold: 20% of max amplitude (generous to catch weak beats)
        const threshold = maxAmp * 0.2;

        state.debugAmplitude = maxAmp;

        if (maxAmp < 0.01) return []; // No meaningful signal

        const peaks = [];
        let lastPeak = -minPeakDist;

        for (let i = 1; i < signal.length - 1; i++) {
            if (
                signal[i] > signal[i - 1] &&
                signal[i] > signal[i + 1] &&
                signal[i] > threshold &&
                (i - lastPeak) >= minPeakDist
            ) {
                peaks.push(i);
                lastPeak = i;
            }
        }

        return peaks;
    }

    /**
     * Validate that peaks represent a real heartbeat by checking:
     * 1. Enough peaks
     * 2. Intervals are consistent (low coefficient of variation)
     * 3. BPM falls in physiological range
     */
    function computeValidBPM(peaks) {
        if (peaks.length < CONFIG.MIN_PEAKS_FOR_BPM) return null;

        const intervals = [];
        for (let i = 1; i < peaks.length; i++) {
            const dt = state.timestamps[peaks[i]] - state.timestamps[peaks[i - 1]];
            if (dt > 0) {
                const bpm = 60000 / dt;
                if (bpm >= CONFIG.MIN_BPM && bpm <= CONFIG.MAX_BPM) {
                    intervals.push(dt);
                }
            }
        }

        if (intervals.length < 2) return null;

        // Check consistency: coefficient of variation should be low
        const mean = intervals.reduce((a, b) => a + b, 0) / intervals.length;
        const variance = intervals.reduce((a, b) => a + (b - mean) ** 2, 0) / intervals.length;
        const cv = Math.sqrt(variance) / mean;

        // If intervals are wildly inconsistent, it's noise not heartbeat
        if (cv > CONFIG.IBI_CONSISTENCY) return null;

        // Use median for final BPM
        const sorted = [...intervals].sort((a, b) => a - b);
        const median = sorted[Math.floor(sorted.length / 2)];
        const bpm = Math.round(60000 / median);

        if (bpm < CONFIG.MIN_BPM || bpm > CONFIG.MAX_BPM) return null;

        state.lastIBI = Math.round(median);
        state.debugPeakCount = peaks.length;

        return bpm;
    }

    // ============================================
    // Helpers
    // ============================================
    function getEffectiveFPS() {
        if (state.timestamps.length < 10) return 30;
        const recent = state.timestamps.slice(-30);
        const elapsed = recent[recent.length - 1] - recent[0];
        if (elapsed <= 0) return 30;
        return Math.round((recent.length - 1) * 1000 / elapsed);
    }

    // ============================================
    // Main Loop
    // ============================================
    function processFrame() {
        if (!state.isRunning) return;

        const now = performance.now();

        // Throttle to ~30fps
        if (now - state.lastSampleTime < 28) {
            state.animationFrameId = requestAnimationFrame(processFrame);
            return;
        }
        state.lastSampleTime = now;
        state.sampleCount++;

        // 1. Sample camera
        const channels = sampleRedChannel();
        if (!channels) {
            state.animationFrameId = requestAnimationFrame(processFrame);
            return;
        }

        // 2. Check finger detection
        state.fingerDetected = isFingerOnCamera(channels);

        // 3. Store raw red value
        state.rawRed.push(channels.red);
        state.timestamps.push(now);

        // 4. Trim buffers
        const fps = getEffectiveFPS();
        const maxSamples = Math.floor(fps * CONFIG.BUFFER_SECONDS);
        while (state.rawRed.length > maxSamples) {
            state.rawRed.shift();
            state.timestamps.shift();
        }

        // 5. Process signal (detrend + smooth)
        let processed = null;
        if (state.rawRed.length > CONFIG.WARMUP_SAMPLES) {
            processed = processSignal();
        }

        // 6. Detect BPM
        const timeSinceStart = now - state.startTime;

        if (processed && state.rawRed.length > CONFIG.WARMUP_SAMPLES && state.fingerDetected) {
            const peaks = findPeaks(processed);
            const bpm = computeValidBPM(peaks);

            if (bpm !== null) {
                state.bpmHistory.push(bpm);
                state.allBpmReadings.push(bpm);

                if (state.bpmHistory.length > CONFIG.BPM_HISTORY_SIZE) {
                    state.bpmHistory.shift();
                }

                // Smoothed current BPM
                state.currentBPM = Math.round(
                    state.bpmHistory.reduce((a, b) => a + b, 0) / state.bpmHistory.length
                );

                // Stats
                state.minBPM = Math.min(state.minBPM, state.currentBPM);
                state.maxBPM = Math.max(state.maxBPM, state.currentBPM);
                state.avgBPM = Math.round(
                    state.allBpmReadings.reduce((a, b) => a + b, 0) / state.allBpmReadings.length
                );

                // Heartbeat animation
                if (peaks.length > 0) {
                    const lastPeakTs = state.timestamps[peaks[peaks.length - 1]];
                    if (lastPeakTs > state.lastBeatTime + 250) { // debounce 250ms
                        state.lastBeatTime = lastPeakTs;
                        triggerHeartbeat();
                    }
                }
            }
        }

        // 7. Signal quality (smoothed)
        const rawQuality = computeSignalQuality();
        state.signalQuality = state.signalQuality * 0.85 + rawQuality * 0.15;

        // 8. Update UI + draw waveform
        updateUI(timeSinceStart);
        drawWaveform();

        state.animationFrameId = requestAnimationFrame(processFrame);
    }

    function computeSignalQuality() {
        if (!state.fingerDetected) return 0.05;

        let q = 0.3; // Base quality for having finger detected

        // Amplitude of AC component
        if (state.debugAmplitude > 0.05) {
            q += Math.min(0.3, state.debugAmplitude * 2);
        }

        // Consistent BPM readings
        if (state.bpmHistory.length >= 3) {
            const recent = state.bpmHistory.slice(-3);
            const mean = recent.reduce((a, b) => a + b, 0) / recent.length;
            const spread = Math.max(...recent) - Math.min(...recent);
            if (spread / mean < 0.1) {
                q += 0.4; // Very consistent
            } else if (spread / mean < 0.2) {
                q += 0.2;
            }
        }

        return Math.min(1, q);
    }

    // ============================================
    // UI
    // ============================================
    function updateUI(timeSinceStart) {
        const timeSeconds = Math.floor(timeSinceStart / 1000);
        DOM.waveformTime.textContent = `${timeSeconds}s`;

        // Signal quality bar
        const qPct = Math.round(state.signalQuality * 100);
        DOM.sqBarFill.style.width = `${qPct}%`;

        const sq = DOM.sqValue;
        if (!state.fingerDetected) {
            sq.textContent = 'No finger detected';
            sq.className = 'sq-value poor';
        } else if (state.signalQuality > 0.6) {
            sq.textContent = 'Excellent';
            sq.className = 'sq-value good';
        } else if (state.signalQuality > 0.3) {
            sq.textContent = 'Good';
            sq.className = 'sq-value fair';
        } else {
            sq.textContent = 'Hold steady...';
            sq.className = 'sq-value poor';
        }

        // BPM
        if (state.currentBPM > 0 && state.fingerDetected) {
            DOM.bpmValue.textContent = state.currentBPM;
            DOM.bpmLabel.textContent = state.signalQuality > 0.5 ? 'Stable reading' : 'Measuring...';
            DOM.bpmLabel.className = 'bpm-label stable';
        } else if (!state.fingerDetected) {
            DOM.bpmValue.textContent = '--';
            DOM.bpmLabel.textContent = 'Cover camera with fingertip';
            DOM.bpmLabel.className = 'bpm-label';
        } else if (state.rawRed.length < CONFIG.WARMUP_SAMPLES) {
            DOM.bpmValue.textContent = '--';
            DOM.bpmLabel.textContent = 'Calibrating...';
            DOM.bpmLabel.className = 'bpm-label';
        } else {
            DOM.bpmValue.textContent = '--';
            DOM.bpmLabel.textContent = 'Analyzing... hold still';
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
    // Waveform
    // ============================================
    function drawWaveform() {
        const canvas = DOM.waveformCanvas;
        const dpr = window.devicePixelRatio || 1;
        const rect = canvas.getBoundingClientRect();

        const cw = Math.round(rect.width * dpr);
        const ch = Math.round(rect.height * dpr);
        if (canvas.width !== cw || canvas.height !== ch) {
            canvas.width = cw;
            canvas.height = ch;
            waveCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
        }

        const w = rect.width;
        const h = rect.height;
        waveCtx.clearRect(0, 0, w, h);

        // Grid
        waveCtx.strokeStyle = 'rgba(255,255,255,0.04)';
        waveCtx.lineWidth = 1;
        for (let i = 1; i < 4; i++) {
            waveCtx.beginPath();
            waveCtx.moveTo(0, (h / 4) * i);
            waveCtx.lineTo(w, (h / 4) * i);
            waveCtx.stroke();
        }
        waveCtx.strokeStyle = 'rgba(255,255,255,0.08)';
        waveCtx.beginPath();
        waveCtx.moveTo(0, h / 2);
        waveCtx.lineTo(w, h / 2);
        waveCtx.stroke();

        const signal = state.detrendedSignal;
        if (!signal || signal.length < 3) return;

        // Show last 5 seconds
        const fps = getEffectiveFPS();
        const displayN = Math.min(signal.length, Math.floor(fps * 5));
        const displayData = signal.slice(-displayN);

        // Auto-scale
        let yMin = Infinity, yMax = -Infinity;
        for (const v of displayData) {
            if (v < yMin) yMin = v;
            if (v > yMax) yMax = v;
        }
        const range = (yMax - yMin) || 1;
        const pad = 0.12;

        const toX = (i) => (i / Math.max(1, displayN - 1)) * w;
        const toY = (v) => {
            const normalized = (v - yMin) / range;  // 0..1
            return h * (1 - pad) - normalized * h * (1 - 2 * pad);
        };

        // Filled area
        const fillGrad = waveCtx.createLinearGradient(0, 0, 0, h);
        fillGrad.addColorStop(0, 'rgba(0,212,216,0.18)');
        fillGrad.addColorStop(0.5, 'rgba(0,212,216,0.05)');
        fillGrad.addColorStop(1, 'rgba(0,212,216,0.0)');

        waveCtx.beginPath();
        waveCtx.moveTo(toX(0), h);
        waveCtx.lineTo(toX(0), toY(displayData[0]));
        for (let i = 1; i < displayData.length; i++) {
            const cx = (toX(i - 1) + toX(i)) / 2;
            const cy = (toY(displayData[i - 1]) + toY(displayData[i])) / 2;
            waveCtx.quadraticCurveTo(toX(i - 1), toY(displayData[i - 1]), cx, cy);
        }
        waveCtx.lineTo(toX(displayData.length - 1), h);
        waveCtx.closePath();
        waveCtx.fillStyle = fillGrad;
        waveCtx.fill();

        // Line
        waveCtx.beginPath();
        waveCtx.moveTo(toX(0), toY(displayData[0]));
        for (let i = 1; i < displayData.length; i++) {
            const cx = (toX(i - 1) + toX(i)) / 2;
            const cy = (toY(displayData[i - 1]) + toY(displayData[i])) / 2;
            waveCtx.quadraticCurveTo(toX(i - 1), toY(displayData[i - 1]), cx, cy);
        }

        const strokeGrad = waveCtx.createLinearGradient(0, 0, w, 0);
        strokeGrad.addColorStop(0, 'rgba(0,212,216,0.25)');
        strokeGrad.addColorStop(0.6, 'rgba(0,212,216,0.85)');
        strokeGrad.addColorStop(1, 'rgba(0,212,216,1)');
        waveCtx.strokeStyle = strokeGrad;
        waveCtx.lineWidth = 2.2;
        waveCtx.lineJoin = 'round';
        waveCtx.lineCap = 'round';
        waveCtx.stroke();

        // End dot with glow
        const ex = toX(displayData.length - 1);
        const ey = toY(displayData[displayData.length - 1]);
        const glow = waveCtx.createRadialGradient(ex, ey, 0, ex, ey, 10);
        glow.addColorStop(0, 'rgba(0,212,216,0.7)');
        glow.addColorStop(1, 'rgba(0,212,216,0)');
        waveCtx.beginPath();
        waveCtx.arc(ex, ey, 10, 0, Math.PI * 2);
        waveCtx.fillStyle = glow;
        waveCtx.fill();
        waveCtx.beginPath();
        waveCtx.arc(ex, ey, 3, 0, Math.PI * 2);
        waveCtx.fillStyle = '#00d4d8';
        waveCtx.fill();
    }

    // ============================================
    // Start / Stop
    // ============================================
    async function startMeasurement() {
        DOM.btnStart.disabled = true;

        const success = await startCamera();
        if (!success) {
            DOM.btnStart.disabled = false;
            return;
        }

        // Reset state
        Object.assign(state, {
            isRunning: true,
            startTime: performance.now(),
            lastSampleTime: 0,
            sampleCount: 0,
            rawRed: [],
            timestamps: [],
            detrendedSignal: [],
            bpmHistory: [],
            allBpmReadings: [],
            currentBPM: 0,
            avgBPM: 0,
            minBPM: Infinity,
            maxBPM: 0,
            lastIBI: 0,
            signalQuality: 0,
            lastBeatTime: 0,
            fingerDetected: false,
            debugRedMean: 0,
            debugRedRatio: 0,
            debugPeakCount: 0,
            debugAmplitude: 0,
        });

        DOM.instructionCard.style.display = 'none';
        DOM.measurementPanel.style.display = 'flex';
        DOM.bpmValue.textContent = '--';
        DOM.bpmLabel.textContent = 'Calibrating...';
        DOM.bpmLabel.className = 'bpm-label';
        DOM.sqBarFill.style.width = '0%';
        DOM.sqValue.textContent = 'Waiting...';
        DOM.sqValue.className = 'sq-value';
        DOM.btnStart.disabled = false;

        state.animationFrameId = requestAnimationFrame(processFrame);
        console.log('[PulseSense] Measurement started');
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

        console.log('[PulseSense] Measurement stopped');
    }

    // ============================================
    // Events
    // ============================================
    DOM.btnStart.addEventListener('click', startMeasurement);
    DOM.btnStop.addEventListener('click', stopMeasurement);

    document.addEventListener('visibilitychange', () => {
        if (document.hidden && state.isRunning) stopMeasurement();
    });

    window.addEventListener('resize', () => {
        if (state.isRunning) drawWaveform();
    });

    console.log('[PulseSense] Ready. Tap "Start Measurement" to begin.');

})();
