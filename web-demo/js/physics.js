/**
 * MechanicsDSL Physics Engine
 */

class PhysicsEngine {
    constructor() {
        this.simulations = {};
    }

    createPendulum(params = {}) {
        const g = params.g || 9.81;
        const l = params.l || 1.0;
        const theta0 = params.theta || 0.5;
        const omega0 = params.omega || 0;

        return {
            type: 'pendulum',
            params: { g, l },
            state: { theta: theta0, omega: omega0 },
            trail: [],
            maxTrail: 200,

            step(dt) {
                const f = (theta) => -g / l * Math.sin(theta);
                const k1_omega = f(this.state.theta);
                const k2_omega = f(this.state.theta + 0.5 * dt * this.state.omega);
                const k3_omega = f(this.state.theta + 0.5 * dt * this.state.omega);
                const k4_omega = f(this.state.theta + dt * this.state.omega);

                this.state.omega += dt * (k1_omega + 2 * k2_omega + 2 * k3_omega + k4_omega) / 6;
                this.state.theta += this.state.omega * dt;
                return this.state;
            },

            energy() {
                return 0.5 * l * l * this.state.omega * this.state.omega + g * l * (1 - Math.cos(this.state.theta));
            },

            render(ctx, width, height) {
                const cx = width / 2, cy = height / 3;
                const scale = Math.min(width, height) * 0.3;
                const x = cx + scale * Math.sin(this.state.theta);
                const y = cy + scale * Math.cos(this.state.theta);

                this.trail.push({ x, y });
                if (this.trail.length > this.maxTrail) this.trail.shift();

                if (this.trail.length > 1) {
                    ctx.beginPath();
                    ctx.moveTo(this.trail[0].x, this.trail[0].y);
                    for (let i = 1; i < this.trail.length; i++) {
                        ctx.strokeStyle = `rgba(99, 102, 241, ${i / this.trail.length * 0.5})`;
                        ctx.lineTo(this.trail[i].x, this.trail[i].y);
                    }
                    ctx.stroke();
                }

                ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(x, y);
                ctx.strokeStyle = '#6366f1'; ctx.lineWidth = 3; ctx.stroke();

                ctx.beginPath(); ctx.arc(cx, cy, 8, 0, Math.PI * 2);
                ctx.fillStyle = '#4f46e5'; ctx.fill();

                ctx.beginPath(); ctx.arc(x, y, 20, 0, Math.PI * 2);
                ctx.fillStyle = '#6366f1'; ctx.fill();
            }
        };
    }

    createDoublePendulum(params = {}) {
        const g = params.g || 9.81;
        const m1 = params.m1 || 1.0, m2 = params.m2 || 1.0;
        const l1 = params.l1 || 1.0, l2 = params.l2 || 1.0;
        const theta1_0 = params.theta1 || 2.5, theta2_0 = params.theta2 || 2.0;
        const omega1_0 = params.omega1 || 0, omega2_0 = params.omega2 || 0;

        return {
            type: 'double-pendulum',
            params: { g, m1, m2, l1, l2 },
            state: { theta1: theta1_0, omega1: omega1_0, theta2: theta2_0, omega2: omega2_0 },
            trail2: [],
            maxTrail: 500,

            step(dt) {
                const { theta1, omega1, theta2, omega2 } = this.state;
                const { g, m1, m2, l1, l2 } = this.params;

                const delta = theta1 - theta2;
                const den1 = (m1 + m2) * l1 - m2 * l1 * Math.cos(delta) * Math.cos(delta);
                const den2 = (l2 / l1) * den1;

                const alpha1 = (m2 * l1 * omega1 * omega1 * Math.sin(delta) * Math.cos(delta)
                    + m2 * g * Math.sin(theta2) * Math.cos(delta) + m2 * l2 * omega2 * omega2 * Math.sin(delta)
                    - (m1 + m2) * g * Math.sin(theta1)) / den1;

                const alpha2 = (-m2 * l2 * omega2 * omega2 * Math.sin(delta) * Math.cos(delta)
                    + (m1 + m2) * g * Math.sin(theta1) * Math.cos(delta) - (m1 + m2) * l1 * omega1 * omega1 * Math.sin(delta)
                    - (m1 + m2) * g * Math.sin(theta2)) / den2;

                this.state.omega1 += alpha1 * dt;
                this.state.omega2 += alpha2 * dt;
                this.state.theta1 += this.state.omega1 * dt;
                this.state.theta2 += this.state.omega2 * dt;
                return this.state;
            },

            energy() {
                const { theta1, omega1, theta2, omega2 } = this.state;
                const { g, m1, m2, l1, l2 } = this.params;
                const y1 = -l1 * Math.cos(theta1);
                const y2 = y1 - l2 * Math.cos(theta2);
                const v1sq = l1 * l1 * omega1 * omega1;
                const v2sq = l1 * l1 * omega1 * omega1 + l2 * l2 * omega2 * omega2 + 2 * l1 * l2 * omega1 * omega2 * Math.cos(theta1 - theta2);
                return 0.5 * m1 * v1sq + 0.5 * m2 * v2sq + m1 * g * y1 + m2 * g * y2 + (m1 + m2) * g * (l1 + l2);
            },

            render(ctx, width, height) {
                const { theta1, theta2 } = this.state;
                const { l1, l2 } = this.params;
                const cx = width / 2, cy = height / 3, scale = Math.min(width, height) * 0.2;

                const x1 = cx + scale * l1 * Math.sin(theta1), y1 = cy + scale * l1 * Math.cos(theta1);
                const x2 = x1 + scale * l2 * Math.sin(theta2), y2 = y1 + scale * l2 * Math.cos(theta2);

                this.trail2.push({ x: x2, y: y2 });
                if (this.trail2.length > this.maxTrail) this.trail2.shift();

                if (this.trail2.length > 1) {
                    ctx.beginPath(); ctx.moveTo(this.trail2[0].x, this.trail2[0].y);
                    for (let i = 1; i < this.trail2.length; i++) {
                        const hue = 260 + (i / this.trail2.length) * 60;
                        ctx.strokeStyle = `hsla(${hue}, 80%, 60%, ${i / this.trail2.length * 0.6})`;
                        ctx.lineTo(this.trail2[i].x, this.trail2[i].y);
                    }
                    ctx.lineWidth = 2; ctx.stroke();
                }

                ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(x1, y1); ctx.lineTo(x2, y2);
                ctx.strokeStyle = '#6366f1'; ctx.lineWidth = 3; ctx.stroke();
                ctx.beginPath(); ctx.arc(cx, cy, 8, 0, Math.PI * 2); ctx.fillStyle = '#4f46e5'; ctx.fill();
                ctx.beginPath(); ctx.arc(x1, y1, 16, 0, Math.PI * 2); ctx.fillStyle = '#06b6d4'; ctx.fill();
                ctx.beginPath(); ctx.arc(x2, y2, 16, 0, Math.PI * 2); ctx.fillStyle = '#f59e0b'; ctx.fill();
            }
        };
    }

    createSpring(params = {}) {
        const k = params.k || 10, m = params.m || 1, b = params.b || 0.1;
        const x0 = params.x || 1.5, v0 = params.v || 0;
        return {
            type: 'spring', params: { k, m, b }, state: { x: x0, v: v0 },
            step(dt) {
                const a = (-this.params.k * this.state.x - this.params.b * this.state.v) / this.params.m;
                this.state.v += a * dt; this.state.x += this.state.v * dt;
                return this.state;
            },
            energy() { return 0.5 * this.params.m * this.state.v ** 2 + 0.5 * this.params.k * this.state.x ** 2; },
            render(ctx, width, height) {
                const cx = width / 2, cy = height / 2, scale = 60;
                const baseX = cx - 150, currentX = baseX + 150 + this.state.x * scale;
                ctx.beginPath(); ctx.moveTo(baseX, cy);
                const coils = 15, springLen = currentX - baseX - 30;
                for (let i = 0; i <= coils; i++) {
                    const t = i / coils;
                    ctx.lineTo(baseX + t * springLen, cy + (i % 2 === 0 ? 20 : -20));
                }
                ctx.lineTo(currentX - 30, cy);
                ctx.strokeStyle = '#6366f1'; ctx.lineWidth = 3; ctx.stroke();
                ctx.fillStyle = '#3f3f50'; ctx.fillRect(baseX - 20, cy - 50, 20, 100);
                ctx.fillStyle = '#06b6d4'; ctx.fillRect(currentX - 30, cy - 25, 50, 50);
            }
        };
    }

    createOrbital(params = {}) {
        const GM = params.GM || 1000, r0 = params.r || 100;
        const v0 = params.v || Math.sqrt(GM / r0) * 0.8;
        return {
            type: 'orbital', params: { GM }, state: { x: r0, y: 0, vx: 0, vy: v0 }, trail: [], maxTrail: 1000,
            step(dt) {
                const { x, y, vx, vy } = this.state;
                const r = Math.sqrt(x * x + y * y);
                const a = -this.params.GM / (r * r * r);
                this.state.vx += a * x * dt; this.state.vy += a * y * dt;
                this.state.x += this.state.vx * dt; this.state.y += this.state.vy * dt;
                return this.state;
            },
            energy() {
                const r = Math.sqrt(this.state.x ** 2 + this.state.y ** 2);
                return 0.5 * (this.state.vx ** 2 + this.state.vy ** 2) - this.params.GM / r;
            },
            render(ctx, width, height) {
                const cx = width / 2, cy = height / 2, s = 1.5;
                const x = cx + this.state.x * s, y = cy + this.state.y * s;
                this.trail.push({ x, y }); if (this.trail.length > this.maxTrail) this.trail.shift();
                if (this.trail.length > 1) {
                    ctx.beginPath(); ctx.moveTo(this.trail[0].x, this.trail[0].y);
                    this.trail.forEach(p => ctx.lineTo(p.x, p.y));
                    ctx.strokeStyle = 'rgba(99,102,241,0.4)'; ctx.lineWidth = 2; ctx.stroke();
                }
                ctx.beginPath(); ctx.arc(cx, cy, 25, 0, Math.PI * 2); ctx.fillStyle = '#f59e0b'; ctx.fill();
                ctx.beginPath(); ctx.arc(x, y, 10, 0, Math.PI * 2); ctx.fillStyle = '#3b82f6'; ctx.fill();
            }
        };
    }

    createSPH(params = {}) {
        const n = params.n || 150, h = 15;
        const particles = [];
        const cols = Math.ceil(Math.sqrt(n));
        for (let i = 0; i < n; i++) {
            particles.push({ x: 50 + (i % cols) * h * 0.8, y: 50 + Math.floor(i / cols) * h * 0.8, vx: 0, vy: 0, density: 1, pressure: 0 });
        }
        return {
            type: 'sph', params: { h, gravity: 0.5, viscosity: 0.1, k: 50 }, particles, width: 400, height: 300,
            step(dt) {
                const { h, gravity, viscosity, k } = this.params;
                const h2 = h * h;
                for (let p of this.particles) {
                    let d = 0;
                    for (let q of this.particles) {
                        const dx = p.x - q.x, dy = p.y - q.y, r2 = dx * dx + dy * dy;
                        if (r2 < h2) d += (h2 - r2) ** 3;
                    }
                    p.density = Math.max(d * 0.00001, 1); p.pressure = k * (p.density - 1);
                }
                for (let p of this.particles) {
                    let ax = 0, ay = gravity;
                    for (let q of this.particles) {
                        if (p === q) continue;
                        const dx = p.x - q.x, dy = p.y - q.y, r = Math.sqrt(dx * dx + dy * dy);
                        if (r > 0 && r < h) {
                            const w = 1 - r / h;
                            ax += (p.pressure + q.pressure) / (2 * q.density) * w * w * dx / r;
                            ay += (p.pressure + q.pressure) / (2 * q.density) * w * w * dy / r;
                            ax += viscosity * w * (q.vx - p.vx) / q.density;
                            ay += viscosity * w * (q.vy - p.vy) / q.density;
                        }
                    }
                    p.vx += ax * dt; p.vy += ay * dt;
                }
                for (let p of this.particles) {
                    p.x += p.vx * dt; p.y += p.vy * dt;
                    if (p.x < 10) { p.x = 10; p.vx *= -0.3; }
                    if (p.x > this.width - 10) { p.x = this.width - 10; p.vx *= -0.3; }
                    if (p.y < 10) { p.y = 10; p.vy *= -0.3; }
                    if (p.y > this.height - 10) { p.y = this.height - 10; p.vy *= -0.3; }
                }
            },
            energy() { return this.particles.reduce((e, p) => e + 0.5 * (p.vx ** 2 + p.vy ** 2) + this.params.gravity * p.y, 0); },
            render(ctx, width, height) {
                const s = Math.min(width / this.width, height / this.height);
                const ox = (width - this.width * s) / 2, oy = (height - this.height * s) / 2;
                ctx.strokeStyle = '#3f3f50'; ctx.lineWidth = 3;
                ctx.strokeRect(ox, oy, this.width * s, this.height * s);
                for (let p of this.particles) {
                    const sp = Math.sqrt(p.vx ** 2 + p.vy ** 2);
                    ctx.beginPath(); ctx.arc(ox + p.x * s, oy + p.y * s, 6 * s, 0, Math.PI * 2);
                    ctx.fillStyle = `hsl(${Math.min(200 + sp * 2, 260)}, 80%, 60%)`; ctx.fill();
                }
            }
        };
    }
}

window.PhysicsEngine = PhysicsEngine;
