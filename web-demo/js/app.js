/**
 * MechanicsDSL Web Demo - Main Application (Enhanced)
 */

document.addEventListener('DOMContentLoaded', () => new MechanicsDSLApp().init());

class MechanicsDSLApp {
    constructor() {
        this.physics = new PhysicsEngine();
        this.simulation = null;
        this.compareSimulation = null;
        this.isRunning = false;
        this.isPaused = false;
        this.compareMode = false;
        this.show3D = false;
        this.speed = 1.0;
        this.time = 0;
        this.frameCount = 0;
        this.lastFpsTime = performance.now();
        this.fps = 60;
        this.theme = 'dark';

        // Three.js
        this.threeScene = null;
        this.threeCamera = null;
        this.threeRenderer = null;
        this.threeMesh = null;

        // Params
        this.params = { g: 9.81, m1: 1.0, m2: 1.0, l1: 1.0, l2: 1.0 };

        // Recording
        this.recordedFrames = [];
    }

    init() {
        this.loadFromURL();
        this.setupTheme();
        this.setupHeroAnimation();
        this.setupCodeEditor();
        this.setupSimulation();
        this.setupSliders();
        this.setupExamples();
        this.setupExport();
        this.setupMiniPreviews();
        this.setupShare();
        this.setupDownload();
        this.setup3D();
        this.loadExample('double-pendulum');
    }

    // ========================================
    // THEME
    // ========================================
    setupTheme() {
        const toggle = document.getElementById('theme-toggle');
        const saved = localStorage.getItem('theme') || 'dark';
        this.setTheme(saved);

        toggle?.addEventListener('click', () => {
            this.setTheme(this.theme === 'dark' ? 'light' : 'dark');
        });
    }

    setTheme(theme) {
        this.theme = theme;
        document.documentElement.setAttribute('data-theme', theme);
        const toggle = document.getElementById('theme-toggle');
        if (toggle) toggle.textContent = theme === 'dark' ? '🌙' : '☀️';
        localStorage.setItem('theme', theme);
    }

    // ========================================
    // URL SHARING
    // ========================================
    loadFromURL() {
        const params = new URLSearchParams(window.location.search);
        if (params.has('example')) {
            setTimeout(() => this.loadExample(params.get('example')), 100);
        }
        if (params.has('theta1')) this.params.theta1 = parseFloat(params.get('theta1'));
        if (params.has('theta2')) this.params.theta2 = parseFloat(params.get('theta2'));
        if (params.has('g')) this.params.g = parseFloat(params.get('g'));
        if (params.has('m1')) this.params.m1 = parseFloat(params.get('m1'));
        if (params.has('m2')) this.params.m2 = parseFloat(params.get('m2'));
    }

    generateShareURL() {
        const url = new URL(window.location.href.split('?')[0]);
        const active = document.querySelector('.example-card.active');
        if (active) url.searchParams.set('example', active.dataset.example);
        url.searchParams.set('g', this.params.g);
        url.searchParams.set('m1', this.params.m1);
        url.searchParams.set('m2', this.params.m2);
        url.searchParams.set('l1', this.params.l1);
        return url.toString();
    }

    setupShare() {
        const shareBtn = document.getElementById('share-btn');
        const modal = document.getElementById('share-modal');
        const closeBtn = document.getElementById('close-modal');
        const urlInput = document.getElementById('share-url');
        const copyBtn = document.getElementById('copy-share-url');

        shareBtn?.addEventListener('click', () => {
            urlInput.value = this.generateShareURL();
            modal.classList.add('active');
        });

        closeBtn?.addEventListener('click', () => modal.classList.remove('active'));
        modal?.addEventListener('click', (e) => {
            if (e.target === modal) modal.classList.remove('active');
        });

        copyBtn?.addEventListener('click', () => {
            navigator.clipboard.writeText(urlInput.value);
            this.showToast('URL copied!');
        });
    }

    // ========================================
    // DOWNLOAD
    // ========================================
    setupDownload() {
        document.getElementById('download-csv-btn')?.addEventListener('click', () => this.downloadCSV());
        document.getElementById('download-gif-btn')?.addEventListener('click', () => this.downloadGIF());
    }

    downloadCSV() {
        if (!this.simulation?.history?.length) {
            this.showToast('No data to export');
            return;
        }

        const headers = Object.keys(this.simulation.history[0]);
        let csv = headers.join(',') + '\n';
        for (const row of this.simulation.history) {
            csv += headers.map(h => row[h]).join(',') + '\n';
        }

        const blob = new Blob([csv], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'simulation_data.csv';
        a.click();
        URL.revokeObjectURL(url);
        this.showToast('CSV downloaded!');
    }

    downloadGIF() {
        this.showToast('Recording... (3 seconds)');
        this.recordedFrames = [];

        const canvas = this.simCanvas;
        const recordInterval = setInterval(() => {
            this.recordedFrames.push(canvas.toDataURL('image/png'));
        }, 100);

        setTimeout(() => {
            clearInterval(recordInterval);
            // Create animated download (simplified - just saves last frame)
            const a = document.createElement('a');
            a.href = this.recordedFrames[this.recordedFrames.length - 1] || canvas.toDataURL();
            a.download = 'simulation_frame.png';
            a.click();
            this.showToast('Frame saved! (Full GIF requires gif.js library)');
        }, 3000);
    }

    // ========================================
    // SLIDERS
    // ========================================
    setupSliders() {
        const sliders = {
            'gravity-slider': { param: 'g', display: 'gravity-value' },
            'm1-slider': { param: 'm1', display: 'm1-value' },
            'm2-slider': { param: 'm2', display: 'm2-value' },
            'l1-slider': { param: 'l1', display: 'l1-value' }
        };

        for (const [id, config] of Object.entries(sliders)) {
            const slider = document.getElementById(id);
            const display = document.getElementById(config.display);

            if (slider && display) {
                slider.value = this.params[config.param];
                display.textContent = this.params[config.param];

                slider.addEventListener('input', () => {
                    this.params[config.param] = parseFloat(slider.value);
                    display.textContent = slider.value;

                    // Update simulation params in real-time
                    if (this.simulation?.params) {
                        this.simulation.params[config.param] = this.params[config.param];
                    }
                    if (this.compareSimulation?.params) {
                        this.compareSimulation.params[config.param] = this.params[config.param];
                    }
                });
            }
        }
    }

    // ========================================
    // 3D VISUALIZATION
    // ========================================
    setup3D() {
        const btn = document.getElementById('3d-btn');
        btn?.addEventListener('click', () => this.toggle3D());
    }

    toggle3D() {
        this.show3D = !this.show3D;
        const container = document.getElementById('three-container');
        const canvas = document.getElementById('sim-canvas');

        if (this.show3D) {
            container.classList.remove('hidden');
            canvas.style.display = 'none';
            document.getElementById('3d-btn').classList.add('active');
            this.init3DScene(container);
        } else {
            container.classList.add('hidden');
            canvas.style.display = 'block';
            document.getElementById('3d-btn').classList.remove('active');
        }
    }

    init3DScene(container) {
        if (this.threeRenderer) return; // Already initialized

        const width = container.clientWidth;
        const height = container.clientHeight;

        this.threeScene = new THREE.Scene();
        this.threeScene.background = new THREE.Color(this.theme === 'dark' ? 0x050508 : 0xe2e8f0);

        this.threeCamera = new THREE.PerspectiveCamera(60, width / height, 0.1, 1000);
        this.threeCamera.position.set(0, 0, 5);

        this.threeRenderer = new THREE.WebGLRenderer({ antialias: true });
        this.threeRenderer.setSize(width, height);
        container.appendChild(this.threeRenderer.domElement);

        // Add lighting
        const ambient = new THREE.AmbientLight(0xffffff, 0.5);
        this.threeScene.add(ambient);
        const directional = new THREE.DirectionalLight(0xffffff, 0.8);
        directional.position.set(5, 5, 5);
        this.threeScene.add(directional);

        // Create a spinning cube/spacecraft
        const geometry = new THREE.BoxGeometry(1, 0.3, 1.5);
        const material = new THREE.MeshPhongMaterial({
            color: 0x6366f1,
            shininess: 100
        });
        this.threeMesh = new THREE.Mesh(geometry, material);
        this.threeScene.add(this.threeMesh);

        // Add coordinate axes
        const axesHelper = new THREE.AxesHelper(2);
        this.threeScene.add(axesHelper);
    }

    render3D() {
        if (!this.threeRenderer || !this.show3D) return;

        // Use pendulum angles to rotate the 3D object
        if (this.simulation && this.threeMesh) {
            const state = this.simulation.state;
            if (state.theta1 !== undefined) {
                this.threeMesh.rotation.x = state.theta1;
                this.threeMesh.rotation.z = state.theta2 || 0;
            } else if (state.theta !== undefined) {
                this.threeMesh.rotation.z = state.theta;
            }
        }

        this.threeRenderer.render(this.threeScene, this.threeCamera);
    }

    // ========================================
    // COMPARE MODE
    // ========================================
    setupCompareMode() {
        document.getElementById('compare-btn')?.addEventListener('click', () => {
            this.toggleCompareMode();
        });
    }

    toggleCompareMode() {
        this.compareMode = !this.compareMode;
        document.getElementById('compare-btn').classList.toggle('active', this.compareMode);

        if (this.compareMode && this.simulation) {
            // Create second simulation with slightly different initial conditions
            const active = document.querySelector('.example-card.active');
            const ex = EXAMPLES[active?.dataset.example || 'double-pendulum'];
            const params = { ...ex.params };

            // Tiny perturbation to show chaos
            if (params.theta1 !== undefined) params.theta1 += 0.001;
            if (params.theta !== undefined) params.theta += 0.001;
            params.color = '#ef4444'; // Red for comparison

            if (params.type === 'double-pendulum') {
                this.compareSimulation = this.physics.createDoublePendulum(params);
            } else if (params.type === 'pendulum') {
                this.compareSimulation = this.physics.createPendulum(params);
            }
        } else {
            this.compareSimulation = null;
        }
    }

    // ========================================
    // HERO ANIMATION
    // ========================================
    setupHeroAnimation() {
        const canvas = document.getElementById('hero-canvas');
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        const resize = () => { canvas.width = window.innerWidth; canvas.height = window.innerHeight; };
        resize(); window.addEventListener('resize', resize);

        const particles = Array.from({ length: 50 }, () => ({
            x: Math.random() * canvas.width, y: Math.random() * canvas.height,
            vx: (Math.random() - 0.5) * 0.5, vy: (Math.random() - 0.5) * 0.5,
            size: Math.random() * 3 + 1, alpha: Math.random() * 0.5 + 0.1
        }));

        const animate = () => {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            for (let i = 0; i < particles.length; i++) {
                for (let j = i + 1; j < particles.length; j++) {
                    const d = Math.hypot(particles[i].x - particles[j].x, particles[i].y - particles[j].y);
                    if (d < 150) {
                        ctx.globalAlpha = (1 - d / 150) * 0.3;
                        ctx.strokeStyle = 'rgba(99,102,241,0.1)';
                        ctx.beginPath(); ctx.moveTo(particles[i].x, particles[i].y);
                        ctx.lineTo(particles[j].x, particles[j].y); ctx.stroke();
                    }
                }
            }
            for (let p of particles) {
                ctx.globalAlpha = p.alpha; ctx.fillStyle = '#6366f1';
                ctx.beginPath(); ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2); ctx.fill();
                p.x += p.vx; p.y += p.vy;
                if (p.x < 0 || p.x > canvas.width) p.vx *= -1;
                if (p.y < 0 || p.y > canvas.height) p.vy *= -1;
            }
            ctx.globalAlpha = 1;
            requestAnimationFrame(animate);
        };
        animate();
    }

    // ========================================
    // CODE EDITOR
    // ========================================
    setupCodeEditor() {
        const input = document.getElementById('code-input');
        const lines = document.getElementById('line-numbers');
        if (!input || !lines) return;
        const update = () => { lines.innerHTML = input.value.split('\n').map((_, i) => i + 1).join('<br>'); };
        input.addEventListener('input', update);
        input.addEventListener('scroll', () => lines.scrollTop = input.scrollTop);
        update();
    }

    // ========================================
    // SIMULATION
    // ========================================
    setupSimulation() {
        this.simCanvas = document.getElementById('sim-canvas');
        this.phaseCanvas = document.getElementById('phase-canvas');
        if (!this.simCanvas) return;

        this.simCtx = this.simCanvas.getContext('2d');
        this.phaseCtx = this.phaseCanvas?.getContext('2d');

        const resize = () => {
            const w = this.simCanvas.parentElement;
            this.simCanvas.width = w.clientWidth;
            this.simCanvas.height = w.clientHeight;

            if (this.phaseCanvas) {
                const pw = this.phaseCanvas.parentElement;
                this.phaseCanvas.width = pw.clientWidth;
                this.phaseCanvas.height = pw.clientHeight;
            }
        };
        resize();

        document.getElementById('run-btn')?.addEventListener('click', () => this.runSimulation());
        document.getElementById('reset-btn')?.addEventListener('click', () => this.resetSimulation());
        document.getElementById('pause-btn')?.addEventListener('click', () => this.togglePause());
        document.getElementById('slow-btn')?.addEventListener('click', () => this.setSpeed(0.25));
        document.getElementById('normal-btn')?.addEventListener('click', () => this.setSpeed(1.0));
        document.getElementById('fast-btn')?.addEventListener('click', () => this.setSpeed(3.0));
        document.getElementById('compare-btn')?.addEventListener('click', () => this.toggleCompareMode());

        // Phase trail storage
        this.phaseTrail = [];
        this.phaseTrail2 = [];
    }

    runSimulation() {
        document.getElementById('canvas-overlay')?.classList.add('hidden');
        const active = document.querySelector('.example-card.active');
        const ex = EXAMPLES[active?.dataset.example || 'double-pendulum'];

        // Merge URL/slider params with example params
        const p = { ...ex.params, ...this.params };

        if (p.type === 'pendulum') this.simulation = this.physics.createPendulum(p);
        else if (p.type === 'double-pendulum') this.simulation = this.physics.createDoublePendulum(p);
        else if (p.type === 'spring') this.simulation = this.physics.createSpring(p);
        else if (p.type === 'orbital') this.simulation = this.physics.createOrbital(p);
        else if (p.type === 'sph') this.simulation = this.physics.createSPH(p);
        else this.simulation = this.physics.createDoublePendulum({});

        // Reset compare sim if in compare mode
        if (this.compareMode) {
            const p2 = { ...p };
            if (p2.theta1 !== undefined) p2.theta1 += 0.001;
            if (p2.theta !== undefined) p2.theta += 0.001;
            p2.color = '#ef4444';
            if (p.type === 'double-pendulum') this.compareSimulation = this.physics.createDoublePendulum(p2);
            else if (p.type === 'pendulum') this.compareSimulation = this.physics.createPendulum(p2);
        }

        this.time = 0;
        this.isRunning = true;
        this.isPaused = false;
        this.phaseTrail = [];
        this.phaseTrail2 = [];
        this.animate();
    }

    resetSimulation() {
        this.isRunning = false;
        this.time = 0;
        this.phaseTrail = [];
        this.phaseTrail2 = [];
        if (this.simCtx) this.simCtx.clearRect(0, 0, this.simCanvas.width, this.simCanvas.height);
        if (this.phaseCtx) this.phaseCtx.clearRect(0, 0, this.phaseCanvas.width, this.phaseCanvas.height);
        document.getElementById('canvas-overlay')?.classList.remove('hidden');
    }

    togglePause() {
        this.isPaused = !this.isPaused;
        const btn = document.getElementById('pause-btn');
        if (btn) btn.textContent = this.isPaused ? '▶' : '⏸';
    }

    setSpeed(s) {
        this.speed = s;
        document.querySelectorAll('.sim-controls .control-btn').forEach(b => {
            if (['slow-btn', 'normal-btn', 'fast-btn'].includes(b.id)) b.classList.remove('active');
        });
        if (s === 0.25) document.getElementById('slow-btn')?.classList.add('active');
        else if (s === 1.0) document.getElementById('normal-btn')?.classList.add('active');
        else if (s === 3.0) document.getElementById('fast-btn')?.classList.add('active');
    }

    animate() {
        if (!this.isRunning) return;

        if (!this.isPaused) {
            const dt = 0.016 * this.speed;
            const substeps = Math.max(1, Math.floor(this.speed * 10));

            if (this.simulation) {
                for (let i = 0; i < substeps; i++) this.simulation.step(dt / substeps);

                // Store phase point
                const pp = this.simulation.getPhasePoint?.();
                if (pp) {
                    this.phaseTrail.push(pp);
                    if (this.phaseTrail.length > 2000) this.phaseTrail.shift();
                }
            }

            if (this.compareSimulation) {
                for (let i = 0; i < substeps; i++) this.compareSimulation.step(dt / substeps);
                const pp2 = this.compareSimulation.getPhasePoint?.();
                if (pp2) {
                    this.phaseTrail2.push(pp2);
                    if (this.phaseTrail2.length > 2000) this.phaseTrail2.shift();
                }
            }

            this.time += dt;
        }

        // Render main canvas
        if (this.simCtx && this.simulation && !this.show3D) {
            const bg = this.theme === 'light' ? '#e2e8f0' : '#050508';
            this.simCtx.fillStyle = bg;
            this.simCtx.fillRect(0, 0, this.simCanvas.width, this.simCanvas.height);
            this.simulation.render(this.simCtx, this.simCanvas.width, this.simCanvas.height, this.theme);

            if (this.compareSimulation) {
                this.compareSimulation.render(this.simCtx, this.simCanvas.width, this.simCanvas.height, this.theme);
            }
        }

        // Render 3D
        this.render3D();

        // Render phase portrait
        this.renderPhasePortrait();

        // Update FPS
        this.frameCount++;
        const now = performance.now();
        if (now - this.lastFpsTime > 1000) {
            this.fps = Math.round(this.frameCount * 1000 / (now - this.lastFpsTime));
            this.frameCount = 0;
            this.lastFpsTime = now;
        }

        // Update info
        const timeEl = document.getElementById('sim-time');
        const energyEl = document.getElementById('sim-energy');
        const fpsEl = document.getElementById('sim-fps');
        const errorEl = document.getElementById('energy-error');
        if (timeEl) timeEl.textContent = this.time.toFixed(2) + ' s';
        if (energyEl) energyEl.textContent = (this.simulation?.energy() || 0).toFixed(2) + ' J';
        if (fpsEl) fpsEl.textContent = this.fps;
        if (errorEl) errorEl.textContent = (this.simulation?.energyError?.() || 0).toFixed(4) + '%';

        requestAnimationFrame(() => this.animate());
    }

    renderPhasePortrait() {
        if (!this.phaseCtx || this.phaseTrail.length < 2) return;

        const ctx = this.phaseCtx;
        const w = this.phaseCanvas.width;
        const h = this.phaseCanvas.height;

        ctx.fillStyle = this.theme === 'light' ? '#e2e8f0' : '#050508';
        ctx.fillRect(0, 0, w, h);

        // Draw axes
        ctx.strokeStyle = this.theme === 'light' ? '#94a3b8' : '#3f3f50';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(0, h / 2); ctx.lineTo(w, h / 2);
        ctx.moveTo(w / 2, 0); ctx.lineTo(w / 2, h);
        ctx.stroke();

        // Auto-scale based on data
        let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
        for (const p of this.phaseTrail) {
            minX = Math.min(minX, p.x); maxX = Math.max(maxX, p.x);
            minY = Math.min(minY, p.y); maxY = Math.max(maxY, p.y);
        }
        const rangeX = Math.max(maxX - minX, 0.1);
        const rangeY = Math.max(maxY - minY, 0.1);
        const scale = Math.min(w / rangeX, h / rangeY) * 0.8;
        const cx = w / 2, cy = h / 2;
        const avgX = (minX + maxX) / 2, avgY = (minY + maxY) / 2;

        // Draw trail
        ctx.beginPath();
        const p0 = this.phaseTrail[0];
        ctx.moveTo(cx + (p0.x - avgX) * scale, cy - (p0.y - avgY) * scale);
        for (let i = 1; i < this.phaseTrail.length; i++) {
            const p = this.phaseTrail[i];
            ctx.lineTo(cx + (p.x - avgX) * scale, cy - (p.y - avgY) * scale);
        }
        ctx.strokeStyle = '#6366f1';
        ctx.lineWidth = 2;
        ctx.stroke();

        // Draw compare trail
        if (this.phaseTrail2.length > 1) {
            ctx.beginPath();
            const q0 = this.phaseTrail2[0];
            ctx.moveTo(cx + (q0.x - avgX) * scale, cy - (q0.y - avgY) * scale);
            for (let i = 1; i < this.phaseTrail2.length; i++) {
                const q = this.phaseTrail2[i];
                ctx.lineTo(cx + (q.x - avgX) * scale, cy - (q.y - avgY) * scale);
            }
            ctx.strokeStyle = '#ef4444';
            ctx.stroke();
        }

        // Draw current point
        const last = this.phaseTrail[this.phaseTrail.length - 1];
        ctx.beginPath();
        ctx.arc(cx + (last.x - avgX) * scale, cy - (last.y - avgY) * scale, 5, 0, Math.PI * 2);
        ctx.fillStyle = '#f59e0b';
        ctx.fill();
    }

    // ========================================
    // EXAMPLES
    // ========================================
    setupExamples() {
        document.querySelectorAll('.example-card').forEach(card => {
            card.addEventListener('click', () => {
                this.loadExample(card.dataset.example);
                document.querySelectorAll('.example-card').forEach(c => c.classList.remove('active'));
                card.classList.add('active');
            });
        });
    }

    loadExample(name) {
        const ex = EXAMPLES[name];
        if (!ex) return;
        const input = document.getElementById('code-input');
        if (input) { input.value = ex.code; input.dispatchEvent(new Event('input')); }
        this.runSimulation();
    }

    setupMiniPreviews() {
        const previews = {
            'preview-pendulum': () => this.physics.createPendulum({ theta: 0.5 }),
            'preview-double': () => this.physics.createDoublePendulum({ theta1: 1.5, theta2: 1.0 }),
            'preview-spring': () => this.physics.createSpring({ x: 1.0 }),
            'preview-orbital': () => this.physics.createOrbital({}),
            'preview-rigid': () => this.physics.createDoublePendulum({ theta1: 0.3, omega2: 5 }),
            'preview-sph': () => this.physics.createSPH({ n: 50 })
        };
        Object.entries(previews).forEach(([id, fn]) => {
            const el = document.getElementById(id);
            if (!el) return;
            const canvas = document.createElement('canvas');
            canvas.width = el.clientWidth || 300; canvas.height = el.clientHeight || 160;
            el.appendChild(canvas);
            const ctx = canvas.getContext('2d'), sim = fn();
            const anim = () => {
                ctx.fillStyle = '#050508'; ctx.fillRect(0, 0, canvas.width, canvas.height);
                for (let i = 0; i < 5; i++) sim.step(0.01);
                sim.render(ctx, canvas.width, canvas.height);
                requestAnimationFrame(anim);
            };
            anim();
        });
    }

    // ========================================
    // EXPORT
    // ========================================
    setupExport() {
        document.querySelectorAll('.export-tab').forEach(tab => {
            tab.addEventListener('click', () => {
                const lang = tab.dataset.lang;
                const pre = document.getElementById('export-preview');
                if (pre) pre.querySelector('code').textContent = CODE_TEMPLATES[lang]?.('pendulum') || '';
                document.querySelectorAll('.export-tab').forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
            });
        });
        document.getElementById('copy-code')?.addEventListener('click', () => {
            navigator.clipboard.writeText(document.getElementById('export-preview')?.textContent || '');
            this.showToast('Code copied!');
        });
        document.getElementById('copy-install')?.addEventListener('click', () => {
            navigator.clipboard.writeText('pip install mechanics-dsl');
            this.showToast('Copied!');
        });
    }

    showToast(message) {
        const container = document.getElementById('toast-container');
        const toast = document.createElement('div');
        toast.className = 'toast';
        toast.textContent = message;
        container.appendChild(toast);
        setTimeout(() => toast.remove(), 2000);
    }
}

// Smooth scroll
document.querySelectorAll('a[href^="#"]').forEach(a => {
    a.addEventListener('click', e => {
        e.preventDefault();
        document.querySelector(a.getAttribute('href'))?.scrollIntoView({ behavior: 'smooth' });
    });
});
