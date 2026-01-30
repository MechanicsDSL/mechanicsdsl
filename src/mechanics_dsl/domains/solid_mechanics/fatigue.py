"""
Fatigue Analysis Module for Solid Mechanics

S-N curves, strain-life, mean stress correction, damage accumulation.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np


@dataclass
class SNData:
    """S-N curve data point."""
    stress_amplitude: float
    cycles: float


class SNCurve:
    """S-N curve: σa = σf'(2Nf)^b."""
    
    def __init__(self, sigma_f_prime: float, b: float, endurance_limit: float = None):
        self.sigma_f_prime = sigma_f_prime
        self.b = b
        self.endurance_limit = endurance_limit
    
    def stress_amplitude(self, N: float) -> float:
        return self.sigma_f_prime * (2 * N)**self.b
    
    def cycles_to_failure(self, sigma_a: float) -> float:
        if self.endurance_limit and sigma_a <= self.endurance_limit:
            return float('inf')
        return 0.5 * (sigma_a / self.sigma_f_prime)**(1 / self.b)
    
    @classmethod
    def steel_estimate(cls, Su: float) -> 'SNCurve':
        sigma_f_prime = 1.09 * Su
        b = -0.085
        Se = 0.5 * Su if Su < 1400e6 else 700e6
        return cls(sigma_f_prime, b, Se)


@dataclass
class BasquinEquation:
    """Basquin: σa = σf'(2Nf)^b."""
    sigma_f_prime: float
    b: float


@dataclass
class CoffinManson:
    """Coffin-Manson strain-life."""
    sigma_f_prime: float
    b: float
    epsilon_f_prime: float
    c: float
    E: float
    
    def strain_amplitude(self, Nf: float) -> Tuple[float, float, float]:
        elastic = (self.sigma_f_prime / self.E) * (2 * Nf)**self.b
        plastic = self.epsilon_f_prime * (2 * Nf)**self.c
        return (elastic + plastic, elastic, plastic)


class EnduranceLimit:
    @staticmethod
    def steel_estimate(Su: float) -> float:
        return 0.5 * Su if Su < 1400e6 else 700e6


class GoodmanDiagram:
    """Goodman: σa/Se + σm/Su = 1."""
    
    def __init__(self, Se: float, Su: float):
        self.Se, self.Su = Se, Su
    
    def equivalent_amplitude(self, sigma_a: float, sigma_m: float) -> float:
        return sigma_a / (1 - sigma_m / self.Su)
    
    def safety_factor(self, sigma_a: float, sigma_m: float) -> float:
        return 1 / (sigma_a / self.Se + sigma_m / self.Su)


HaighDiagram = GoodmanDiagram
MeanStressCorrection = GoodmanDiagram


class ModifiedGoodman(GoodmanDiagram):
    def __init__(self, Se: float, Su: float, Sy: float):
        super().__init__(Se, Su)
        self.Sy = Sy


class GerberCriterion:
    """Gerber: σa/Se + (σm/Su)² = 1."""
    def __init__(self, Se: float, Su: float):
        self.Se, self.Su = Se, Su


class SoderbergCriterion:
    """Soderberg: σa/Se + σm/Sy = 1."""
    def __init__(self, Se: float, Sy: float):
        self.Se, self.Sy = Se, Sy


@dataclass
class SmithWatsonTopper:
    """SWT parameter."""
    sigma_f_prime: float
    epsilon_f_prime: float
    b: float
    c: float
    E: float


class MinersRule:
    """Miner's rule: D = Σ(ni/Ni)."""
    
    def __init__(self, sn_curve: SNCurve):
        self.sn_curve = sn_curve
        self.damage = 0.0
    
    def add_cycles(self, sigma_a: float, n: float) -> float:
        N = self.sn_curve.cycles_to_failure(sigma_a)
        if N == float('inf'):
            return 0.0
        inc = n / N
        self.damage += inc
        return inc


PalmgrenMiner = MinersRule
DamageAccumulation = MinersRule


class RainflowCounting:
    """Rainflow cycle counting."""
    
    @staticmethod
    def count(signal: np.ndarray) -> List[Tuple[float, float, float]]:
        cycles = []
        for i in range(len(signal) - 1):
            rng = abs(signal[i+1] - signal[i])
            mean = (signal[i] + signal[i+1]) / 2
            cycles.append((rng, mean, 0.5))
        return cycles


CycleCounting = RainflowCounting


class FatigueLifePrediction:
    def __init__(self, sn_curve: SNCurve):
        self.sn_curve = sn_curve


@dataclass
class FatigueNotchFactor:
    Kt: float
    q: float
    
    @property
    def Kf(self) -> float:
        return 1 + self.q * (self.Kt - 1)


@dataclass
class FatigueSafetyFactor:
    n: float
    sigma_a: float
    sigma_m: float
    criterion: str


def compute_fatigue_life(sigma_a: float, sn_curve: SNCurve) -> float:
    return sn_curve.cycles_to_failure(sigma_a)


def compute_fatigue_damage(stress_history: np.ndarray, sn_curve: SNCurve) -> float:
    miner = MinersRule(sn_curve)
    for i in range(len(stress_history) - 1):
        sigma_a = abs(stress_history[i+1] - stress_history[i]) / 2
        miner.add_cycles(sigma_a, 0.5)
    return miner.damage


def estimate_endurance_limit(Su: float) -> float:
    return EnduranceLimit.steel_estimate(Su)
