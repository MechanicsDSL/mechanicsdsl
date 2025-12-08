document.addEventListener('DOMContentLoaded', () => new MechanicsDSLApp().init());

class MechanicsDSLApp {
    constructor() {
        this.physics = new PhysicsEngine();
        this.simulation = null;
        this.isRunning = false;
        this.isPaused = false;
        this.speed = 1.0;
        this.time = 0;
        this.frameCount = 0;
        this.lastFpsTime = performance.now();
        this.fps = 60;
    }

    init() {
        this.setupHeroAnimation();
        this.setupCodeEditor();
        this.setupSimulation();
        this.setupExamples();
        this.setupExport();
        this.setupMiniPreviews();
        this.loadExample('double-pendulum');
    }

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

    setupCodeEditor() {
        const input = document.getElementById('code-input');
        const lines = document.getElementById('line-numbers');
        if (!input || !lines) return;
        const update = () => { lines.innerHTML = input.value.split('\n').map((_, i) => i + 1).join('<br>'); };
        input.addEventListener('input', update);
        input.addEventListener('scroll', () => lines.scrollTop = input.scrollTop);
        update();
    }

    setupSimulation() {
        this.simCanvas = document.getElementById('sim-canvas');
        if (!this.simCanvas) return;
        this.simCtx = this.simCanvas.getContext('2d');
        const resize = () => {
            const w = this.simCanvas.parentElement;
            this.simCanvas.width = w.clientWidth;
            this.simCanvas.height = w.clientHeight;
        };
        resize();

        document.getElementById('run-btn')?.addEventListener('click', () => this.runSimulation());
        document.getElementById('reset-btn')?.addEventListener('click', () => this.resetSimulation());
        document.getElementById('pause-btn')?.addEventListener('click', () => this.togglePause());
        document.getElementById('slow-btn')?.addEventListener('click', () => this.setSpeed(0.25));
        document.getElementById('normal-btn')?.addEventListener('click', () => this.setSpeed(1.0));
        document.getElementById('fast-btn')?.addEventListener('click', () => this.setSpeed(3.0));
    }

    runSimulation() {
        document.getElementById('canvas-overlay')?.classList.add('hidden');
        const active = document.querySelector('.example-card.active');
        const ex = EXAMPLES[active?.dataset.example || 'double-pendulum'];
        const p = ex.params;

        if (p.type === 'pendulum') this.simulation = this.physics.createPendulum(p);
        else if (p.type === 'double-pendulum') this.simulation = this.physics.createDoublePendulum(p);
        else if (p.type === 'spring') this.simulation = this.physics.createSpring(p);
        else if (p.type === 'orbital') this.simulation = this.physics.createOrbital(p);
        else if (p.type === 'sph') this.simulation = this.physics.createSPH(p);
        else this.simulation = this.physics.createDoublePendulum({});

        this.time = 0; this.isRunning = true; this.isPaused = false;
        this.animate();
    }

    resetSimulation() {
        this.isRunning = false; this.time = 0;
        if (this.simCtx) this.simCtx.clearRect(0, 0, this.simCanvas.width, this.simCanvas.height);
        document.getElementById('canvas-overlay')?.classList.remove('hidden');
    }

    togglePause() {
        this.isPaused = !this.isPaused;
        const btn = document.getElementById('pause-btn');
        if (btn) btn.textContent = this.isPaused ? '▶' : '⏸';
    }

    setSpeed(s) {
        this.speed = s;
        document.querySelectorAll('.control-btn').forEach(b => b.classList.remove('active'));
        if (s === 0.25) document.getElementById('slow-btn')?.classList.add('active');
        else if (s === 1.0) document.getElementById('normal-btn')?.classList.add('active');
        else if (s === 3.0) document.getElementById('fast-btn')?.classList.add('active');
    }

    animate() {
        if (!this.isRunning) return;
        if (!this.isPaused && this.simulation) {
            const dt = 0.016 * this.speed;
            for (let i = 0; i < Math.max(1, this.speed * 10); i++) this.simulation.step(dt / Math.max(1, this.speed * 10));
            this.time += dt;
        }
        if (this.simCtx && this.simulation) {
            this.simCtx.fillStyle = '#050508';
            this.simCtx.fillRect(0, 0, this.simCanvas.width, this.simCanvas.height);
            this.simulation.render(this.simCtx, this.simCanvas.width, this.simCanvas.height);
        }
        this.frameCount++;
        const now = performance.now();
        if (now - this.lastFpsTime > 1000) {
            this.fps = Math.round(this.frameCount * 1000 / (now - this.lastFpsTime));
            this.frameCount = 0; this.lastFpsTime = now;
        }
        document.getElementById('sim-time').textContent = this.time.toFixed(2) + ' s';
        document.getElementById('sim-energy').textContent = (this.simulation?.energy() || 0).toFixed(2) + ' J';
        document.getElementById('sim-fps').textContent = this.fps;
        requestAnimationFrame(() => this.animate());
    }

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
        });
        document.getElementById('copy-install')?.addEventListener('click', () => {
            navigator.clipboard.writeText('pip install mechanics-dsl');
        });
    }
}

document.querySelectorAll('a[href^="#"]').forEach(a => {
    a.addEventListener('click', e => {
        e.preventDefault();
        document.querySelector(a.getAttribute('href'))?.scrollIntoView({ behavior: 'smooth' });
    });
});
