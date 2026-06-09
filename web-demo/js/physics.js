/**
 * MechanicsDSL Physics Engine - Enhanced
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
            initialEnergy: null,
            trail: [],
            phaseTrail: [],
            maxTrail: 200,
            history: [],

            step(dt) {
                const { g, l } = this.params;
                const f = (theta) => -g / l * Math.sin(theta);

                // RK4
                const k1_theta = this.state.omega;
                const k1_omega = f(this.state.theta);
                const k2_theta = this.state.omega + 0.5 * dt * k1_omega;
                const k2_omega = f(this.state.theta + 0.5 * dt * k1_theta);
                const k3_theta = this.state.omega + 0.5 * dt * k2_omega;
                const k3_omega = f(this.state.theta + 0.5 * dt * k2_theta);
                const k4_theta = this.state.omega + dt * k3_omega;
                const k4_omega = f(this.state.theta + dt * k3_theta);

                this.state.theta += dt * (k1_theta + 2 * k2_theta + 2 * k3_theta + k4_theta) / 6;
                this.state.omega += dt * (k1_omega + 2 * k2_omega + 2 * k3_omega + k4_omega) / 6;

                // Track initial energy
                if (this.initialEnergy === null) this.initialEnergy = this.energy();

                // Store history for export
                this.history.push({ ...this.state, energy: this.energy() });
                if (this.history.length > 10000) this.history.shift();

                return this.state;
            },

            energy() {
                const { g, l } = this.params;
                return 0.5 * l * l * this.state.omega * this.state.omega + g * l * (1 - Math.cos(this.state.theta));
            },

            energyError() {
                if (!this.initialEnergy) return 0;
                return Math.abs((this.energy() - this.initialEnergy) / this.initialEnergy) * 100;
            },

            getPhasePoint() {
                return { x: this.state.theta, y: this.state.omega };
            },

            render(ctx, width, height, theme = 'dark') {
                const cx = width / 2, cy = height / 3;
                const scale = Math.min(width, height) * 0.3;
                const x = cx + scale * Math.sin(this.state.theta);
                const y = cy + scale * Math.cos(this.state.theta);

                this.trail.push({ x, y });
                if (this.trail.length > this.maxTrail) this.trail.shift();

                const colors = theme === 'light' ?
                    { trail: 'rgba(79, 70, 229, ', rod: '#4f46e5', pivot: '#4338ca', bob: '#6366f1' } :
                    { trail: 'rgba(99, 102, 241, ', rod: '#6366f1', pivot: '#4f46e5', bob: '#6366f1' };

                if (this.trail.length > 1) {
                    ctx.beginPath();
                    ctx.moveTo(this.trail[0].x, this.trail[0].y);
                    for (let i = 1; i < this.trail.length; i++) {
                        ctx.strokeStyle = colors.trail + (i / this.trail.length * 0.5) + ')';
                        ctx.lineTo(this.trail[i].x, this.trail[i].y);
                    }
                    ctx.stroke();
                }

                ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(x, y);
                ctx.strokeStyle = colors.rod; ctx.lineWidth = 3; ctx.stroke();

                ctx.beginPath(); ctx.arc(cx, cy, 8, 0, Math.PI * 2);
                ctx.fillStyle = colors.pivot; ctx.fill();

                ctx.beginPath(); ctx.arc(x, y, 20, 0, Math.PI * 2);
                ctx.fillStyle = colors.bob; ctx.fill();
            }
        };
    }

    createDoublePendulum(params = {}) {
        const g = params.g || 9.81;
        const m1 = params.m1 || 1.0, m2 = params.m2 || 1.0;
        const l1 = params.l1 || 1.0, l2 = params.l2 || 1.0;
        const theta1_0 = params.theta1 !== undefined ? params.theta1 : 2.5;
        const theta2_0 = params.theta2 !== undefined ? params.theta2 : 2.0;
        const omega1_0 = params.omega1 || 0, omega2_0 = params.omega2 || 0;

        return {
            type: 'double-pendulum',
            params: { g, m1, m2, l1, l2 },
            state: { theta1: theta1_0, omega1: omega1_0, theta2: theta2_0, omega2: omega2_0 },
            initialEnergy: null,
            trail2: [],
            maxTrail: 500,
            history: [],
            color: params.color || '#f59e0b',

            step(dt) {
                const { g, m1, m2, l1, l2 } = this.params;

                // Compute accelerations given state
                const accel = (t1, o1, t2, o2) => {
                    const delta = t1 - t2;
                    const cosDelta = Math.cos(delta);
                    const sinDelta = Math.sin(delta);
                    const den1 = (m1 + m2) * l1 - m2 * l1 * cosDelta * cosDelta;
                    const den2 = (l2 / l1) * den1;

                    const a1 = (m2 * l1 * o1 * o1 * sinDelta * cosDelta
                        + m2 * g * Math.sin(t2) * cosDelta
                        + m2 * l2 * o2 * o2 * sinDelta
                        - (m1 + m2) * g * Math.sin(t1)) / den1;

                    const a2 = (-m2 * l2 * o2 * o2 * sinDelta * cosDelta
                        + (m1 + m2) * g * Math.sin(t1) * cosDelta
                        - (m1 + m2) * l1 * o1 * o1 * sinDelta
                        - (m1 + m2) * g * Math.sin(t2)) / den2;

                    return [a1, a2];
                };

                // RK4 integration
                const { theta1, omega1, theta2, omega2 } = this.state;

                // k1
                const [a1_1, a2_1] = accel(theta1, omega1, theta2, omega2);
                const k1 = [omega1, a1_1, omega2, a2_1];

                // k2
                const [a1_2, a2_2] = accel(
                    theta1 + 0.5 * dt * k1[0], omega1 + 0.5 * dt * k1[1],
                    theta2 + 0.5 * dt * k1[2], omega2 + 0.5 * dt * k1[3]
                );
                const k2 = [omega1 + 0.5 * dt * k1[1], a1_2, omega2 + 0.5 * dt * k1[3], a2_2];

                // k3
                const [a1_3, a2_3] = accel(
                    theta1 + 0.5 * dt * k2[0], omega1 + 0.5 * dt * k2[1],
                    theta2 + 0.5 * dt * k2[2], omega2 + 0.5 * dt * k2[3]
                );
                const k3 = [omega1 + 0.5 * dt * k2[1], a1_3, omega2 + 0.5 * dt * k2[3], a2_3];

                // k4
                const [a1_4, a2_4] = accel(
                    theta1 + dt * k3[0], omega1 + dt * k3[1],
                    theta2 + dt * k3[2], omega2 + dt * k3[3]
                );
                const k4 = [omega1 + dt * k3[1], a1_4, omega2 + dt * k3[3], a2_4];

                // Update state
                this.state.theta1 += dt * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6;
                this.state.omega1 += dt * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6;
                this.state.theta2 += dt * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]) / 6;
                this.state.omega2 += dt * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3]) / 6;

                if (this.initialEnergy === null) this.initialEnergy = this.energy();

                this.history.push({ ...this.state, energy: this.energy() });
                if (this.history.length > 10000) this.history.shift();

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

            energyError() {
                if (!this.initialEnergy) return 0;
                return Math.abs((this.energy() - this.initialEnergy) / this.initialEnergy) * 100;
            },

            getPhasePoint() {
                return { x: this.state.theta2, y: this.state.omega2 };
            },

            render(ctx, width, height, theme = 'dark') {
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
                ctx.beginPath(); ctx.arc(x2, y2, 16, 0, Math.PI * 2); ctx.fillStyle = this.color; ctx.fill();
            }
        };
    }

    createSpring(params = {}) {
        const k = params.k || 10, m = params.m || 1, b = params.b || 0.1;
        const x0 = params.x || 1.5, v0 = params.v || 0;
        return {
            type: 'spring', params: { k, m, b }, state: { x: x0, v: v0 },
            initialEnergy: null, history: [],
            step(dt) {
                const a = (-this.params.k * this.state.x - this.params.b * this.state.v) / this.params.m;
                this.state.v += a * dt; this.state.x += this.state.v * dt;
                if (this.initialEnergy === null) this.initialEnergy = this.energy();
                this.history.push({ ...this.state, energy: this.energy() });
                if (this.history.length > 10000) this.history.shift();
                return this.state;
            },
            energy() { return 0.5 * this.params.m * this.state.v ** 2 + 0.5 * this.params.k * this.state.x ** 2; },
            energyError() {
                if (!this.initialEnergy) return 0;
                return Math.abs((this.energy() - this.initialEnergy) / this.initialEnergy) * 100;
            },
            getPhasePoint() { return { x: this.state.x, y: this.state.v }; },
            render(ctx, width, height, theme = 'dark') {
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
                ctx.fillStyle = theme === 'light' ? '#64748b' : '#3f3f50';
                ctx.fillRect(baseX - 20, cy - 50, 20, 100);
                ctx.fillStyle = '#06b6d4'; ctx.fillRect(currentX - 30, cy - 25, 50, 50);
            }
        };
    }

    createOrbital(params = {}) {
        const GM = params.GM || 1000, r0 = params.r || 100;
        const v0 = params.v || Math.sqrt(GM / r0) * 0.8;
        return {
            type: 'orbital', params: { GM }, state: { x: r0, y: 0, vx: 0, vy: v0 },
            initialEnergy: null, trail: [], maxTrail: 1000, history: [],
            step(dt) {
                const { x, y, vx, vy } = this.state;
                const r = Math.sqrt(x * x + y * y);
                const a = -this.params.GM / (r * r * r);
                this.state.vx += a * x * dt; this.state.vy += a * y * dt;
                this.state.x += this.state.vx * dt; this.state.y += this.state.vy * dt;
                if (this.initialEnergy === null) this.initialEnergy = this.energy();
                this.history.push({ ...this.state, energy: this.energy() });
                if (this.history.length > 10000) this.history.shift();
                return this.state;
            },
            energy() {
                const r = Math.sqrt(this.state.x ** 2 + this.state.y ** 2);
                return 0.5 * (this.state.vx ** 2 + this.state.vy ** 2) - this.params.GM / r;
            },
            energyError() {
                if (!this.initialEnergy) return 0;
                return Math.abs((this.energy() - this.initialEnergy) / this.initialEnergy) * 100;
            },
            getPhasePoint() { return { x: this.state.x, y: this.state.vx }; },
            render(ctx, width, height, theme = 'dark') {
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

    createRigidBody(params = {}) {
        // Asymmetric top: Euler's equations + quaternion attitude.
        // Defaults illustrate the Dzhanibekov effect (intermediate-axis theorem):
        // an object spun about its intermediate principal axis will periodically flip.
        const I1 = params.I1 || 1.0;        // smallest principal moment
        const I2 = params.I2 || 2.0;        // intermediate
        const I3 = params.I3 || 3.0;        // largest
        const w1_0 = params.wx ?? 0.05;     // small perturbation
        const w2_0 = params.wy ?? 3.0;      // dominant spin about intermediate axis
        const w3_0 = params.wz ?? 0.0;
        // Unit quaternion (w, x, y, z); start aligned with world frame.
        const q0 = { w: 1, x: 0, y: 0, z: 0 };

        const qmul = (a, b) => ({
            w: a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
            x: a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
            y: a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
            z: a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w
        });
        const qnorm = (q) => {
            const n = Math.sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z) || 1;
            return { w: q.w / n, x: q.x / n, y: q.y / n, z: q.z / n };
        };
        const qrotate = (q, v) => {
            // Rodrigues form: q * (0,v) * q^-1, expanded
            const { w, x, y, z } = q;
            const t2 = 2 * (y * v[2] - z * v[1]);
            const t3 = 2 * (z * v[0] - x * v[2]);
            const t4 = 2 * (x * v[1] - y * v[0]);
            return [
                v[0] + w * t2 + (y * t4 - z * t3),
                v[1] + w * t3 + (z * t2 - x * t4),
                v[2] + w * t4 + (x * t3 - y * t2)
            ];
        };

        return {
            type: 'rigid-body',
            params: { I1, I2, I3 },
            state: { q: { ...q0 }, w1: w1_0, w2: w2_0, w3: w3_0 },
            initialEnergy: null,
            angularTrail: [],
            maxTrail: 400,
            history: [],
            color: params.color || null,  // honored by render() for compare mode

            // Derivatives of (q, ω) — Euler's equations in body frame plus quaternion kinematics.
            deriv(q, w1, w2, w3) {
                const { I1, I2, I3 } = this.params;
                const dw1 = (I2 - I3) * w2 * w3 / I1;
                const dw2 = (I3 - I1) * w3 * w1 / I2;
                const dw3 = (I1 - I2) * w1 * w2 / I3;
                // dq/dt = 0.5 * q ⊗ (0, ω_body)
                const dq = qmul(q, { w: 0, x: w1 * 0.5, y: w2 * 0.5, z: w3 * 0.5 });
                return { dq, dw1, dw2, dw3 };
            },

            step(dt) {
                const add = (q, dq, s) => ({
                    w: q.w + dq.w * s, x: q.x + dq.x * s,
                    y: q.y + dq.y * s, z: q.z + dq.z * s
                });

                const { q, w1, w2, w3 } = this.state;

                const k1 = this.deriv(q, w1, w2, w3);
                const k2 = this.deriv(
                    add(q, k1.dq, dt / 2),
                    w1 + k1.dw1 * dt / 2, w2 + k1.dw2 * dt / 2, w3 + k1.dw3 * dt / 2
                );
                const k3 = this.deriv(
                    add(q, k2.dq, dt / 2),
                    w1 + k2.dw1 * dt / 2, w2 + k2.dw2 * dt / 2, w3 + k2.dw3 * dt / 2
                );
                const k4 = this.deriv(
                    add(q, k3.dq, dt),
                    w1 + k3.dw1 * dt, w2 + k3.dw2 * dt, w3 + k3.dw3 * dt
                );

                const sixth = dt / 6;
                const q_new = {
                    w: q.w + sixth * (k1.dq.w + 2 * k2.dq.w + 2 * k3.dq.w + k4.dq.w),
                    x: q.x + sixth * (k1.dq.x + 2 * k2.dq.x + 2 * k3.dq.x + k4.dq.x),
                    y: q.y + sixth * (k1.dq.y + 2 * k2.dq.y + 2 * k3.dq.y + k4.dq.y),
                    z: q.z + sixth * (k1.dq.z + 2 * k2.dq.z + 2 * k3.dq.z + k4.dq.z)
                };
                this.state.q = qnorm(q_new);
                this.state.w1 += sixth * (k1.dw1 + 2 * k2.dw1 + 2 * k3.dw1 + k4.dw1);
                this.state.w2 += sixth * (k1.dw2 + 2 * k2.dw2 + 2 * k3.dw2 + k4.dw2);
                this.state.w3 += sixth * (k1.dw3 + 2 * k2.dw3 + 2 * k3.dw3 + k4.dw3);

                if (this.initialEnergy === null) this.initialEnergy = this.energy();

                this.angularTrail.push({ x: this.state.w1, y: this.state.w3 });
                if (this.angularTrail.length > this.maxTrail) this.angularTrail.shift();

                this.history.push({
                    qw: this.state.q.w, qx: this.state.q.x, qy: this.state.q.y, qz: this.state.q.z,
                    w1: this.state.w1, w2: this.state.w2, w3: this.state.w3,
                    energy: this.energy()
                });
                if (this.history.length > 10000) this.history.shift();

                return this.state;
            },

            energy() {
                // Rotational kinetic energy in the body frame: ½ Σ Iᵢ ωᵢ²
                const { I1, I2, I3 } = this.params;
                const { w1, w2, w3 } = this.state;
                return 0.5 * (I1 * w1 * w1 + I2 * w2 * w2 + I3 * w3 * w3);
            },

            energyError() {
                if (!this.initialEnergy) return 0;
                return Math.abs((this.energy() - this.initialEnergy) / this.initialEnergy) * 100;
            },

            getPhasePoint() {
                // ω₁ vs ω₃ — both flip sign during a Dzhanibekov event
                return { x: this.state.w1, y: this.state.w3 };
            },

            render(ctx, width, height, theme = 'dark') {
                const cx = width / 2, cy = height / 2;
                const scale = Math.min(width, height) * 0.22;
                const camDist = 4;

                // Half-extents — short along the largest-inertia axis, long along the
                // smallest, so the rendered shape matches the principal-axis ordering.
                const hx = 1.3, hy = 0.6, hz = 0.3;
                const verts = [
                    [-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
                    [-hx, -hy,  hz], [hx, -hy,  hz], [hx, hy,  hz], [-hx, hy,  hz]
                ];

                // Rotate then perspective-project.
                const rotated = verts.map(v => qrotate(this.state.q, v));
                const project = (p) => {
                    const z = p[2] + camDist;
                    const k = camDist / z;
                    return { x: cx + p[0] * scale * k, y: cy + p[1] * scale * k, z };
                };
                const proj = rotated.map(project);

                const base = this.color || '#5b8def';
                // For comparison sims, render all faces in the override color so the
                // two bodies are visually distinct; otherwise vary per face for depth.
                const faces = this.color ? [
                    { idx: [0, 1, 2, 3], color: base }, { idx: [4, 5, 6, 7], color: base },
                    { idx: [0, 1, 5, 4], color: base }, { idx: [2, 3, 7, 6], color: base },
                    { idx: [0, 3, 7, 4], color: base }, { idx: [1, 2, 6, 5], color: base }
                ] : [
                    { idx: [0, 1, 2, 3], color: '#5b8def' }, // -z
                    { idx: [4, 5, 6, 7], color: '#3a6fe0' }, //  z
                    { idx: [0, 1, 5, 4], color: '#8b6dff' }, // -y
                    { idx: [2, 3, 7, 6], color: '#6b8de0' }, //  y
                    { idx: [0, 3, 7, 4], color: '#7559e6' }, // -x
                    { idx: [1, 2, 6, 5], color: '#67e8f9' }  //  x
                ];
                // Painter's algorithm: draw far faces first.
                faces.sort((a, b) => {
                    const za = a.idx.reduce((s, i) => s + rotated[i][2], 0);
                    const zb = b.idx.reduce((s, i) => s + rotated[i][2], 0);
                    return za - zb;
                });

                ctx.lineWidth = 1.5;
                for (const f of faces) {
                    const pts = f.idx.map(i => proj[i]);
                    ctx.beginPath();
                    ctx.moveTo(pts[0].x, pts[0].y);
                    for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x, pts[i].y);
                    ctx.closePath();
                    ctx.fillStyle = f.color + 'cc';
                    ctx.strokeStyle = theme === 'light' ? '#1e293b' : '#e7ecf3';
                    ctx.fill();
                    ctx.stroke();
                }

                // Spin axis indicator: project (ω₁, ω₂, ω₃) into world frame.
                const omegaWorld = qrotate(this.state.q, [this.state.w1, this.state.w2, this.state.w3]);
                const wmag = Math.sqrt(omegaWorld[0] ** 2 + omegaWorld[1] ** 2 + omegaWorld[2] ** 2) || 1;
                const len = hx + 0.5;
                const tip  = project([ omegaWorld[0] / wmag * len,  omegaWorld[1] / wmag * len,  omegaWorld[2] / wmag * len]);
                const tail = project([-omegaWorld[0] / wmag * len, -omegaWorld[1] / wmag * len, -omegaWorld[2] / wmag * len]);
                ctx.strokeStyle = '#f59e0b';
                ctx.lineWidth = 2;
                ctx.setLineDash([6, 4]);
                ctx.beginPath();
                ctx.moveTo(tail.x, tail.y);
                ctx.lineTo(tip.x, tip.y);
                ctx.stroke();
                ctx.setLineDash([]);
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
            initialEnergy: null, history: [],
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
                if (this.initialEnergy === null) this.initialEnergy = this.energy();
            },
            energy() { return this.particles.reduce((e, p) => e + 0.5 * (p.vx ** 2 + p.vy ** 2) + this.params.gravity * p.y, 0); },
            energyError() { return 0; }, // SPH doesn't conserve energy well
            getPhasePoint() { return { x: this.particles[0]?.x || 0, y: this.particles[0]?.vy || 0 }; },
            render(ctx, width, height, theme = 'dark') {
                const s = Math.min(width / this.width, height / this.height);
                const ox = (width - this.width * s) / 2, oy = (height - this.height * s) / 2;
                ctx.strokeStyle = theme === 'light' ? '#94a3b8' : '#3f3f50'; ctx.lineWidth = 3;
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
