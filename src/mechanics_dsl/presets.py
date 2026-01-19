"""
MechanicsDSL Presets Library

Built-in physics system definitions that users can import.
"""

# Classical Mechanics Presets
PENDULUM = r"""\system{simple_pendulum}
\defvar{theta}{Angle from vertical}{rad}
\parameter{m}{1.0}{kg}
\parameter{l}{1.0}{m}
\parameter{g}{9.81}{m/s^2}
\lagrangian{\frac{1}{2}*m*l^2*\dot{theta}^2 - m*g*l*(1-\cos{theta})}
\initial{theta=0.5, theta_dot=0}
"""

DOUBLE_PENDULUM = r"""\system{double_pendulum}
\defvar{theta1}{Upper arm angle}{rad}
\defvar{theta2}{Lower arm angle}{rad}
\parameter{m1}{1.0}{kg}
\parameter{m2}{1.0}{kg}
\parameter{l1}{1.0}{m}
\parameter{l2}{1.0}{m}
\parameter{g}{9.81}{m/s^2}
\lagrangian{
    \frac{1}{2}*(m1+m2)*l1^2*\dot{theta1}^2
    + \frac{1}{2}*m2*l2^2*\dot{theta2}^2
    + m2*l1*l2*\dot{theta1}*\dot{theta2}*\cos{theta1-theta2}
    - (m1+m2)*g*l1*(1-\cos{theta1})
    - m2*g*l2*(1-\cos{theta2})
}
\initial{theta1=2.5, theta2=2.0}
"""

SPRING_MASS = r"""\system{spring_mass}
\defvar{x}{Displacement from equilibrium}{m}
\parameter{m}{1.0}{kg}
\parameter{k}{10.0}{N/m}
\lagrangian{\frac{1}{2}*m*\dot{x}^2 - \frac{1}{2}*k*x^2}
\initial{x=1.0, x_dot=0}
"""

DAMPED_OSCILLATOR = r"""\system{damped_oscillator}
\defvar{x}{Displacement}{m}
\parameter{m}{1.0}{kg}
\parameter{k}{10.0}{N/m}
\parameter{b}{0.5}{N*s/m}
\lagrangian{\frac{1}{2}*m*\dot{x}^2 - \frac{1}{2}*k*x^2}
\force{-b*\dot{x}}
\initial{x=1.0, x_dot=0}
"""

PROJECTILE = r"""\system{projectile}
\defvar{x}{Horizontal position}{m}
\defvar{y}{Vertical position}{m}
\parameter{m}{1.0}{kg}
\parameter{g}{9.81}{m/s^2}
\lagrangian{\frac{1}{2}*m*(\dot{x}^2 + \dot{y}^2) - m*g*y}
\initial{x=0, y=0, x_dot=10, y_dot=10}
"""

COUPLED_OSCILLATORS = r"""\system{coupled_oscillators}
\defvar{x1}{Position of mass 1}{m}
\defvar{x2}{Position of mass 2}{m}
\parameter{m1}{1.0}{kg}
\parameter{m2}{1.0}{kg}
\parameter{k1}{10.0}{N/m}
\parameter{k2}{5.0}{N/m}
\parameter{k3}{10.0}{N/m}
\lagrangian{
    \frac{1}{2}*m1*\dot{x1}^2 + \frac{1}{2}*m2*\dot{x2}^2
    - \frac{1}{2}*k1*x1^2 - \frac{1}{2}*k2*(x2-x1)^2 - \frac{1}{2}*k3*x2^2
}
\initial{x1=1.0, x2=0}
"""

# Celestial Mechanics Presets
KEPLER_ORBIT = r"""\system{kepler_orbit}
\defvar{r}{Radial distance}{m}
\defvar{phi}{Azimuthal angle}{rad}
\parameter{m}{1.0}{kg}
\parameter{M}{1.989e30}{kg}
\parameter{G}{6.674e-11}{N*m^2/kg^2}
\lagrangian{
    \frac{1}{2}*m*(\dot{r}^2 + r^2*\dot{phi}^2) + G*M*m/r
}
\initial{r=1.496e11, phi=0, r_dot=0, phi_dot=1.991e-7}
"""

THREE_BODY_FIGURE8 = r"""\system{figure8_orbit}
\defvar{x1}{x position body 1}{m}
\defvar{y1}{y position body 1}{m}
\defvar{x2}{x position body 2}{m}
\defvar{y2}{y position body 2}{m}
\defvar{x3}{x position body 3}{m}
\defvar{y3}{y position body 3}{m}
\parameter{m}{1.0}{kg}
\parameter{G}{1.0}{N*m^2/kg^2}
\lagrangian{
    0.5*m*(\dot{x1}^2+\dot{y1}^2+\dot{x2}^2+\dot{y2}^2+\dot{x3}^2+\dot{y3}^2)
    + G*m^2/\sqrt{(x1-x2)^2+(y1-y2)^2}
    + G*m^2/\sqrt{(x2-x3)^2+(y2-y3)^2}
    + G*m^2/\sqrt{(x1-x3)^2+(y1-y3)^2}
}
\initial{x1=0.97, y1=0, x2=-0.97, y2=0, x3=0, y3=0}
\initial{x1_dot=-0.466, y1_dot=-0.433, x2_dot=-0.466, y2_dot=-0.433}
\initial{x3_dot=0.932, y3_dot=0.866}
"""

# Registry of all presets
PRESETS = {
    # Classical
    'pendulum': PENDULUM,
    'simple_pendulum': PENDULUM,
    'double_pendulum': DOUBLE_PENDULUM,
    'spring': SPRING_MASS,
    'spring_mass': SPRING_MASS,
    'harmonic_oscillator': SPRING_MASS,
    'damped': DAMPED_OSCILLATOR,
    'damped_oscillator': DAMPED_OSCILLATOR,
    'projectile': PROJECTILE,
    'coupled': COUPLED_OSCILLATORS,
    'coupled_oscillators': COUPLED_OSCILLATORS,
    # Celestial
    'kepler': KEPLER_ORBIT,
    'orbit': KEPLER_ORBIT,
    'figure8': THREE_BODY_FIGURE8,
    'three_body': THREE_BODY_FIGURE8,
}


def get_preset(name: str) -> str:
    """
    Get a preset DSL definition by name.
    
    Args:
        name: Preset name (case-insensitive)
        
    Returns:
        DSL source code for the preset
        
    Raises:
        KeyError: If preset not found
    """
    key = name.lower().strip()
    if key not in PRESETS:
        available = ', '.join(sorted(set(PRESETS.keys())))
        raise KeyError(f"Preset '{name}' not found. Available: {available}")
    return PRESETS[key]


def list_presets() -> list[str]:
    """Return list of available preset names."""
    return sorted(set(PRESETS.keys()))


__all__ = ['PRESETS', 'get_preset', 'list_presets']
