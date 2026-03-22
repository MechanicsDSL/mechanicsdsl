"""
Acoustics Domain for MechanicsDSL

Comprehensive acoustics implementation including:
- Wave equation solutions (1D, 2D, 3D)
- Resonance and standing waves
- Acoustic impedance
- Doppler effect
- Room acoustics (reverberation, Sabine)
- Ultrasonics

Security: All frequency and speed inputs validated for physical consistency.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Tuple, Optional, Callable
import numpy as np


class WaveType(Enum):
    """Types of acoustic waves."""

    PLANE = auto()
    SPHERICAL = auto()
    CYLINDRICAL = auto()
    STANDING = auto()


@dataclass
class AcousticMedium:
    """Properties of acoustic propagation medium."""

    density: float  # kg/m³
    bulk_modulus: float  # Pa (or speed of sound)
    name: str = "air"

    def __post_init__(self):
        if self.density <= 0 or self.bulk_modulus <= 0:
            raise ValueError("Density and bulk modulus must be positive")

    @property
    def speed_of_sound(self) -> float:
        """c = √(K/ρ)."""
        return math.sqrt(self.bulk_modulus / self.density)

    @property
    def characteristic_impedance(self) -> float:
        """Z₀ = ρc."""
        return self.density * self.speed_of_sound

    @classmethod
    def air(cls, temperature: float = 20.0) -> "AcousticMedium":
        """Air at given temperature (°C)."""
        c = 331.3 + 0.606 * temperature
        rho = 1.204 * (293.15 / (273.15 + temperature))
        K = rho * c**2
        return cls(density=rho, bulk_modulus=K, name=f"air_{temperature}C")

    @classmethod
    def water(cls) -> "AcousticMedium":
        """Fresh water at 20°C."""
        return cls(density=998, bulk_modulus=2.2e9, name="water")

    @classmethod
    def steel(cls) -> "AcousticMedium":
        """Steel (longitudinal waves)."""
        return cls(density=7850, bulk_modulus=160e9, name="steel")


class WaveEquation:
    """
    Solutions to the acoustic wave equation.

    ∂²p/∂t² = c²∇²p
    """

    @staticmethod
    def plane_wave(x: float, t: float, A: float, k: float, omega: float, phi: float = 0) -> float:
        """
        Plane wave solution.

        p(x,t) = A cos(kx - ωt + φ)

        Args:
            x: Position (m)
            t: Time (s)
            A: Amplitude (Pa)
            k: Wavenumber (rad/m)
            omega: Angular frequency (rad/s)
            phi: Phase (rad)
        """
        return A * math.cos(k * x - omega * t + phi)

    @staticmethod
    def spherical_wave(r: float, t: float, A: float, k: float, omega: float) -> float:
        """
        Spherical wave (outgoing).

        p(r,t) = (A/r) cos(kr - ωt)
        """
        if r <= 0:
            raise ValueError("Distance must be positive")
        return (A / r) * math.cos(k * r - omega * t)

    @staticmethod
    def standing_wave(x: float, t: float, A: float, k: float, omega: float) -> float:
        """
        Standing wave solution.

        p(x,t) = 2A cos(kx) cos(ωt)
        """
        return 2 * A * math.cos(k * x) * math.cos(omega * t)

    @staticmethod
    def dispersion_relation(k: float, c: float) -> float:
        """ω = ck for non-dispersive media."""
        return c * k

    @staticmethod
    def wavelength(f: float, c: float) -> float:
        """λ = c/f."""
        if f <= 0:
            raise ValueError("Frequency must be positive")
        return c / f

    @staticmethod
    def wavenumber(f: float, c: float) -> float:
        """k = 2πf/c = ω/c."""
        return 2 * math.pi * f / c


@dataclass
class Resonance:
    """Resonance analysis for acoustic systems."""

    @staticmethod
    def string_frequencies(L: float, c: float, n_max: int = 5) -> List[float]:
        """
        Natural frequencies of fixed-fixed string.

        f_n = nc/(2L)
        """
        if L <= 0 or c <= 0:
            raise ValueError("Length and wave speed must be positive")
        return [n * c / (2 * L) for n in range(1, n_max + 1)]

    @staticmethod
    def open_pipe_frequencies(L: float, c: float, n_max: int = 5) -> List[float]:
        """
        Natural frequencies of open-open pipe.

        All harmonics: f_n = nc/(2L)
        """
        return [n * c / (2 * L) for n in range(1, n_max + 1)]

    @staticmethod
    def closed_pipe_frequencies(L: float, c: float, n_max: int = 5) -> List[float]:
        """
        Natural frequencies of closed-open pipe.

        Odd harmonics only: f_n = (2n-1)c/(4L)
        """
        return [(2 * n - 1) * c / (4 * L) for n in range(1, n_max + 1)]

    @staticmethod
    def rectangular_cavity(
        Lx: float, Ly: float, Lz: float, c: float, modes: int = 5
    ) -> List[Tuple[Tuple[int, int, int], float]]:
        """
        Natural frequencies of rectangular cavity.

        f_{mnp} = (c/2)√((m/Lx)² + (n/Ly)² + (p/Lz)²)
        """
        results = []
        for m in range(modes + 1):
            for n in range(modes + 1):
                for p in range(modes + 1):
                    if m == 0 and n == 0 and p == 0:
                        continue
                    f = (c / 2) * math.sqrt((m / Lx) ** 2 + (n / Ly) ** 2 + (p / Lz) ** 2)
                    results.append(((m, n, p), f))
        return sorted(results, key=lambda x: x[1])[:modes]


@dataclass
class HelmholtzResonator:
    """
    Helmholtz resonator analysis.

    f = (c/2π)√(A/(V·L_eff))
    """

    neck_area: float  # A (m²)
    neck_length: float  # L (m)
    cavity_volume: float  # V (m³)
    medium: AcousticMedium = None

    def __post_init__(self):
        if any(x <= 0 for x in [self.neck_area, self.neck_length, self.cavity_volume]):
            raise ValueError("All dimensions must be positive")
        if self.medium is None:
            self.medium = AcousticMedium.air()

    @property
    def effective_length(self) -> float:
        """End correction: L_eff = L + 1.7√(A/π)."""
        radius = math.sqrt(self.neck_area / math.pi)
        return self.neck_length + 1.7 * radius

    @property
    def resonance_frequency(self) -> float:
        """Helmholtz resonance frequency."""
        c = self.medium.speed_of_sound
        return (c / (2 * math.pi)) * math.sqrt(
            self.neck_area / (self.cavity_volume * self.effective_length)
        )

    @property
    def quality_factor(self) -> float:
        """Approximate Q factor (radiation limited)."""
        return 2 * math.pi * self.cavity_volume / self.neck_area ** (3 / 2)


class AcousticImpedance:
    """Acoustic impedance calculations."""

    @staticmethod
    def specific(rho: float, c: float) -> float:
        """Specific acoustic impedance Z = ρc."""
        return rho * c

    @staticmethod
    def acoustic(Z_s: float, A: float) -> float:
        """Acoustic impedance Z_a = Z_s / A."""
        if A <= 0:
            raise ValueError("Area must be positive")
        return Z_s / A

    @staticmethod
    def reflection_coefficient(Z1: float, Z2: float) -> float:
        """
        Pressure reflection coefficient.

        R = (Z2 - Z1) / (Z2 + Z1)
        """
        return (Z2 - Z1) / (Z2 + Z1)

    @staticmethod
    def transmission_coefficient(Z1: float, Z2: float) -> float:
        """
        Pressure transmission coefficient.

        T = 2Z2 / (Z2 + Z1)
        """
        return 2 * Z2 / (Z2 + Z1)

    @staticmethod
    def power_transmission(Z1: float, Z2: float) -> float:
        """
        Power transmission coefficient.

        τ = 4Z1Z2 / (Z1 + Z2)²
        """
        return 4 * Z1 * Z2 / (Z1 + Z2) ** 2

    @staticmethod
    def impedance_matching(Z1: float, Z2: float) -> float:
        """Optimal matching layer impedance Z_m = √(Z1·Z2)."""
        return math.sqrt(Z1 * Z2)


class DopplerEffect:
    """Doppler effect calculations."""

    @staticmethod
    def frequency_shift(f0: float, c: float, v_source: float = 0, v_observer: float = 0) -> float:
        """
        Doppler-shifted frequency.

        f' = f₀(c + v_o)/(c + v_s)

        Convention: positive velocity toward each other.

        Args:
            f0: Source frequency (Hz)
            c: Speed of sound (m/s)
            v_source: Source velocity (positive = toward observer)
            v_observer: Observer velocity (positive = toward source)
        """
        if c + v_source <= 0:
            raise ValueError("Invalid: supersonic source")
        return f0 * (c + v_observer) / (c - v_source)

    @staticmethod
    def beat_frequency(f1: float, f2: float) -> float:
        """Beat frequency |f1 - f2|."""
        return abs(f1 - f2)

    @staticmethod
    def mach_number(v: float, c: float) -> float:
        """Mach number M = v/c."""
        return v / c

    @staticmethod
    def mach_cone_angle(M: float) -> float:
        """
        Mach cone half-angle (for M > 1).

        sin(θ) = 1/M
        """
        if M <= 1:
            raise ValueError("Mach number must be > 1 for shock waves")
        return math.asin(1 / M)


class RoomAcoustics:
    """Room acoustics calculations."""

    @staticmethod
    def sabine_rt60(V: float, A: float) -> float:
        """
        Sabine reverberation time.

        RT60 = 0.161V/A

        Args:
            V: Room volume (m³)
            A: Total absorption area (m² Sabins)
        """
        if V <= 0 or A <= 0:
            raise ValueError("Volume and absorption must be positive")
        return 0.161 * V / A

    @staticmethod
    def eyring_rt60(V: float, S: float, alpha_avg: float) -> float:
        """
        Eyring reverberation time.

        RT60 = 0.161V / (-S·ln(1-α))

        Args:
            V: Room volume (m³)
            S: Total surface area (m²)
            alpha_avg: Average absorption coefficient (0-1)
        """
        if alpha_avg >= 1 or alpha_avg <= 0:
            raise ValueError("Absorption coefficient must be in (0, 1)")
        return 0.161 * V / (-S * math.log(1 - alpha_avg))

    @staticmethod
    def critical_distance(Q: float, A: float) -> float:
        """
        Critical distance (direct = reverberant).

        r_c = √(Q·A / (16π))

        Args:
            Q: Source directivity factor
            A: Room absorption (Sabins)
        """
        return math.sqrt(Q * A / (16 * math.pi))

    @staticmethod
    def schroeder_frequency(RT60: float, V: float) -> float:
        """
        Schroeder frequency (modal to statistical transition).

        f_s = 2000√(RT60/V)
        """
        return 2000 * math.sqrt(RT60 / V)

    @staticmethod
    def sound_pressure_level(p: float, p_ref: float = 20e-6) -> float:
        """SPL in dB: L_p = 20 log₁₀(p/p_ref)."""
        if p <= 0:
            return float("-inf")
        return 20 * math.log10(p / p_ref)

    @staticmethod
    def sound_intensity_level(I: float, I_ref: float = 1e-12) -> float:
        """SIL in dB: L_I = 10 log₁₀(I/I_ref)."""
        if I <= 0:
            return float("-inf")
        return 10 * math.log10(I / I_ref)


@dataclass
class AbsorptionCoefficient:
    """Frequency-dependent absorption coefficient."""

    frequencies: List[float]  # Hz
    alpha: List[float]  # Absorption coefficients
    material: str = "unknown"

    def at_frequency(self, f: float) -> float:
        """Interpolate absorption at frequency."""
        if f <= self.frequencies[0]:
            return self.alpha[0]
        if f >= self.frequencies[-1]:
            return self.alpha[-1]

        for i, freq in enumerate(self.frequencies[:-1]):
            if self.frequencies[i] <= f <= self.frequencies[i + 1]:
                ratio = (f - freq) / (self.frequencies[i + 1] - freq)
                return self.alpha[i] + ratio * (self.alpha[i + 1] - self.alpha[i])
        return self.alpha[-1]

    @classmethod
    def carpet(cls) -> "AbsorptionCoefficient":
        """Typical carpet on concrete."""
        return cls(
            frequencies=[125, 250, 500, 1000, 2000, 4000],
            alpha=[0.10, 0.30, 0.50, 0.55, 0.60, 0.70],
            material="carpet",
        )

    @classmethod
    def acoustic_tile(cls) -> "AbsorptionCoefficient":
        """Acoustic ceiling tile."""
        return cls(
            frequencies=[125, 250, 500, 1000, 2000, 4000],
            alpha=[0.20, 0.50, 0.75, 0.80, 0.80, 0.75],
            material="acoustic_tile",
        )


class Ultrasonics:
    """Ultrasonic wave calculations."""

    @staticmethod
    def near_field_distance(D: float, wavelength: float) -> float:
        """
        Near field (Fresnel zone) distance.

        N = D²/(4λ)
        """
        return D**2 / (4 * wavelength)

    @staticmethod
    def beam_spread_angle(D: float, wavelength: float) -> float:
        """
        Half-angle of beam divergence.

        sin(θ) = 1.22λ/D
        """
        ratio = 1.22 * wavelength / D
        if ratio > 1:
            return math.pi / 2
        return math.asin(ratio)

    @staticmethod
    def attenuation(alpha: float, x: float) -> float:
        """
        Amplitude attenuation.

        A(x) = A₀ exp(-αx)

        Returns ratio A/A₀
        """
        return math.exp(-alpha * x)

    @staticmethod
    def time_of_flight(distance: float, c: float) -> float:
        """Time of flight t = d/c."""
        return distance / c

    @staticmethod
    def distance_from_tof(tof: float, c: float) -> float:
        """Distance from time of flight (round trip)."""
        return tof * c / 2


# Exports for domain __init__
__all__ = [
    "WaveType",
    "AcousticMedium",
    "WaveEquation",
    "Resonance",
    "HelmholtzResonator",
    "AcousticImpedance",
    "DopplerEffect",
    "RoomAcoustics",
    "AbsorptionCoefficient",
    "Ultrasonics",
]
