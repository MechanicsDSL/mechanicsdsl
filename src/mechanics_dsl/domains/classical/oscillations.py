"""
Small Oscillations and Normal Mode Analysis

This module implements:
- Mass and stiffness matrix extraction
- Normal mode computation
- Modal decomposition for forced response
- Coupled oscillator analysis

For small oscillations about equilibrium:
    L ≈ (1/2) * q̇ᵀ * M * q̇ - (1/2) * qᵀ * K * q

The equations of motion become:
    M * q̈ + K * q = 0

Normal modes satisfy the generalized eigenvalue problem:
    K * φ = ω² * M * φ
"""
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import sympy as sp
import numpy as np
from numpy.linalg import eig, inv

from ...utils import logger


@dataclass
class NormalMode:
    """
    Represents a single normal mode of oscillation.
    
    Attributes:
        frequency: Natural frequency ω (rad/s)
        period: Period T = 2π/ω (s)
        mode_shape: Normalized eigenvector φ
        participation_factor: How much each DOF participates
        mode_index: Index of this mode (0 = lowest frequency)
    """
    frequency: float
    period: float
    mode_shape: np.ndarray
    participation_factor: Optional[np.ndarray] = None
    mode_index: int = 0
    
    def __repr__(self) -> str:
        return f"NormalMode(ω={self.frequency:.4f} rad/s, T={self.period:.4f} s)"


@dataclass
class ModalAnalysisResult:
    """
    Complete result of modal analysis.
    
    Attributes:
        modes: List of NormalMode objects (sorted by frequency)
        mass_matrix: Mass matrix M
        stiffness_matrix: Stiffness matrix K
        modal_matrix: Matrix of mode shapes (columns are modes)
        coordinates: Coordinate names
    """
    modes: List[NormalMode]
    mass_matrix: np.ndarray
    stiffness_matrix: np.ndarray
    modal_matrix: np.ndarray
    coordinates: List[str]
    
    def get_frequencies(self) -> np.ndarray:
        """Get array of natural frequencies."""
        return np.array([mode.frequency for mode in self.modes])
    
    def get_periods(self) -> np.ndarray:
        """Get array of periods."""
        return np.array([mode.period for mode in self.modes])
    
    def __repr__(self) -> str:
        freqs = self.get_frequencies()
        return f"ModalAnalysisResult(n_modes={len(self.modes)}, frequencies={freqs})"


class NormalModeAnalyzer:
    """
    Normal mode analysis for coupled oscillator systems.
    
    Extracts mass and stiffness matrices from a Lagrangian and
    computes normal modes via the generalized eigenvalue problem.
    
    Example:
        >>> analyzer = NormalModeAnalyzer()
        >>> # Define Lagrangian for coupled pendulums
        >>> L = 0.5*m*(theta1_dot**2 + theta2_dot**2) - 0.5*k*(theta1**2 + theta2**2 + (theta1-theta2)**2)
        >>> result = analyzer.analyze(L, ['theta1', 'theta2'])
        >>> print(result.get_frequencies())
    """
    
    def __init__(self):
        self._symbol_cache: Dict[str, sp.Symbol] = {}
    
    def get_symbol(self, name: str, **assumptions) -> sp.Symbol:
        """Get or create a symbol with caching."""
        if name not in self._symbol_cache:
            default_assumptions = {'real': True}
            default_assumptions.update(assumptions)
            self._symbol_cache[name] = sp.Symbol(name, **default_assumptions)
        return self._symbol_cache[name]
    
    def extract_mass_matrix(self, lagrangian: sp.Expr, 
                             coordinates: List[str]) -> sp.Matrix:
        """
        Extract mass matrix M from kinetic energy.
        
        For T = (1/2) * q̇ᵀ * M * q̇:
            Mᵢⱼ = ∂²T/∂q̇ᵢ∂q̇ⱼ
        
        Evaluated at equilibrium (q = 0, q̇ = 0).
        
        Args:
            lagrangian: Lagrangian L(q, q̇)
            coordinates: List of generalized coordinates
            
        Returns:
            Symbolic mass matrix
        """
        n = len(coordinates)
        M = sp.zeros(n, n)
        
        q_dot_symbols = [self.get_symbol(f"{q}_dot") for q in coordinates]
        
        for i, qi_dot in enumerate(q_dot_symbols):
            for j, qj_dot in enumerate(q_dot_symbols):
                # Mᵢⱼ = ∂²L/∂q̇ᵢ∂q̇ⱼ (for L = T - V, only T depends on q̇)
                M[i, j] = sp.diff(lagrangian, qi_dot, qj_dot)
        
        # Evaluate at equilibrium (all q = 0, q̇ = 0)
        q_symbols = [self.get_symbol(q) for q in coordinates]
        subs_dict = {q: 0 for q in q_symbols}
        subs_dict.update({qd: 0 for qd in q_dot_symbols})
        
        M = M.subs(subs_dict)
        
        return M
    
    def extract_stiffness_matrix(self, lagrangian: sp.Expr,
                                   coordinates: List[str]) -> sp.Matrix:
        """
        Extract stiffness matrix K from potential energy.
        
        For V = (1/2) * qᵀ * K * q:
            Kᵢⱼ = ∂²V/∂qᵢ∂qⱼ = -∂²L/∂qᵢ∂qⱼ (for L = T - V)
        
        Evaluated at equilibrium (q = 0, q̇ = 0).
        
        Args:
            lagrangian: Lagrangian L(q, q̇)
            coordinates: List of generalized coordinates
            
        Returns:
            Symbolic stiffness matrix
        """
        n = len(coordinates)
        K = sp.zeros(n, n)
        
        q_symbols = [self.get_symbol(q) for q in coordinates]
        q_dot_symbols = [self.get_symbol(f"{q}_dot") for q in coordinates]
        
        for i, qi in enumerate(q_symbols):
            for j, qj in enumerate(q_symbols):
                # Kᵢⱼ = -∂²L/∂qᵢ∂qⱼ (L = T - V, so -∂²L/∂q² = ∂²V/∂q²)
                K[i, j] = -sp.diff(lagrangian, qi, qj)
        
        # Evaluate at equilibrium
        subs_dict = {q: 0 for q in q_symbols}
        subs_dict.update({qd: 0 for qd in q_dot_symbols})
        
        K = K.subs(subs_dict)
        
        return K
    
    def compute_normal_modes(self, M: np.ndarray, K: np.ndarray) -> List[NormalMode]:
        """
        Compute normal modes from mass and stiffness matrices.
        
        Solves the generalized eigenvalue problem:
            K * φ = ω² * M * φ
        
        Args:
            M: Mass matrix (n x n)
            K: Stiffness matrix (n x n)
            
        Returns:
            List of NormalMode objects sorted by frequency
        """
        try:
            M_inv = inv(M)
        except np.linalg.LinAlgError:
            logger.error("Mass matrix is singular")
            return []
        
        # Solve eigenvalue problem for M⁻¹K
        eigenvalues, eigenvectors = eig(M_inv @ K)
        
        modes = []
        for i, (lam, phi) in enumerate(zip(eigenvalues, eigenvectors.T)):
            # ω² = λ
            omega_squared = np.real(lam)
            
            if omega_squared < 0:
                logger.warning(f"Mode {i} has negative ω² = {omega_squared}, indicating instability")
                omega = 0.0
                period = float('inf')
            elif omega_squared < 1e-10:
                omega = 0.0
                period = float('inf')
            else:
                omega = np.sqrt(omega_squared)
                period = 2 * np.pi / omega
            
            # Normalize mode shape
            phi_normalized = np.real(phi) / np.linalg.norm(np.real(phi))
            
            # Participation factor: how much mass is in each DOF
            participation = np.abs(phi_normalized) ** 2
            participation = participation / np.sum(participation)
            
            modes.append(NormalMode(
                frequency=omega,
                period=period,
                mode_shape=phi_normalized,
                participation_factor=participation,
                mode_index=i
            ))
        
        # Sort by frequency (lowest first)
        modes.sort(key=lambda m: m.frequency)
        for i, mode in enumerate(modes):
            mode.mode_index = i
        
        return modes
    
    def analyze(self, lagrangian: sp.Expr, coordinates: List[str],
                parameters: Optional[Dict[str, float]] = None) -> ModalAnalysisResult:
        """
        Perform complete modal analysis on a Lagrangian.
        
        Args:
            lagrangian: System Lagrangian L(q, q̇)
            coordinates: List of generalized coordinates
            parameters: Optional parameter values for numerical evaluation
            
        Returns:
            ModalAnalysisResult with modes and matrices
        """
        # Extract symbolic matrices
        M_sym = self.extract_mass_matrix(lagrangian, coordinates)
        K_sym = self.extract_stiffness_matrix(lagrangian, coordinates)
        
        # Substitute parameters if provided
        if parameters:
            for name, value in parameters.items():
                sym = sp.Symbol(name)
                M_sym = M_sym.subs(sym, value)
                K_sym = K_sym.subs(sym, value)
        
        # Convert to numerical
        try:
            M = np.array(M_sym.evalf(), dtype=float)
            K = np.array(K_sym.evalf(), dtype=float)
        except Exception as e:
            logger.error(f"Could not evaluate matrices: {e}")
            return ModalAnalysisResult(
                modes=[],
                mass_matrix=np.array([]),
                stiffness_matrix=np.array([]),
                modal_matrix=np.array([]),
                coordinates=coordinates
            )
        
        # Compute modes
        modes = self.compute_normal_modes(M, K)
        
        # Build modal matrix (columns are mode shapes)
        if modes:
            modal_matrix = np.column_stack([m.mode_shape for m in modes])
        else:
            modal_matrix = np.array([])
        
        return ModalAnalysisResult(
            modes=modes,
            mass_matrix=M,
            stiffness_matrix=K,
            modal_matrix=modal_matrix,
            coordinates=coordinates
        )
    
    def modal_decomposition(self, M: np.ndarray, K: np.ndarray,
                             F: np.ndarray, omega_drive: float) -> np.ndarray:
        """
        Compute steady-state response to harmonic forcing via modal decomposition.
        
        For M*q̈ + K*q = F*cos(ωt), the response is:
            q(t) = Σᵢ φᵢ * Aᵢ * cos(ωt)
        
        where Aᵢ = (φᵢᵀ * F) / (mᵢ * (ωᵢ² - ω²))
        
        Args:
            M: Mass matrix
            K: Stiffness matrix
            F: Force amplitude vector
            omega_drive: Driving frequency
            
        Returns:
            Amplitude vector q₀ such that q(t) = q₀ * cos(ωt)
        """
        modes = self.compute_normal_modes(M, K)
        
        if not modes:
            return np.zeros(M.shape[0])
        
        # Modal mass: mᵢ = φᵢᵀ * M * φᵢ
        # Modal stiffness: kᵢ = φᵢᵀ * K * φᵢ = mᵢ * ωᵢ²
        # Modal force: fᵢ = φᵢᵀ * F
        
        q_response = np.zeros(M.shape[0])
        
        for mode in modes:
            phi = mode.mode_shape
            omega_n = mode.frequency
            
            modal_mass = phi @ M @ phi
            modal_force = phi @ F
            
            # Avoid division by zero at resonance
            if abs(omega_n**2 - omega_drive**2) < 1e-10:
                logger.warning(f"Resonance at mode {mode.mode_index}")
                amplitude = 1e10 * np.sign(modal_force)  # Large response
            else:
                amplitude = modal_force / (modal_mass * (omega_n**2 - omega_drive**2))
            
            q_response += phi * amplitude
        
        return q_response


def extract_mass_matrix(lagrangian: sp.Expr, coordinates: List[str]) -> sp.Matrix:
    """
    Convenience function to extract mass matrix from Lagrangian.
    
    Args:
        lagrangian: Lagrangian L(q, q̇)
        coordinates: List of generalized coordinates
        
    Returns:
        Symbolic mass matrix
    """
    analyzer = NormalModeAnalyzer()
    return analyzer.extract_mass_matrix(lagrangian, coordinates)


def extract_stiffness_matrix(lagrangian: sp.Expr, coordinates: List[str]) -> sp.Matrix:
    """
    Convenience function to extract stiffness matrix from Lagrangian.
    
    Args:
        lagrangian: Lagrangian L(q, q̇)
        coordinates: List of generalized coordinates
        
    Returns:
        Symbolic stiffness matrix
    """
    analyzer = NormalModeAnalyzer()
    return analyzer.extract_stiffness_matrix(lagrangian, coordinates)


def compute_normal_modes(lagrangian: sp.Expr, coordinates: List[str],
                          parameters: Optional[Dict[str, float]] = None) -> ModalAnalysisResult:
    """
    Convenience function for complete modal analysis.
    
    Args:
        lagrangian: System Lagrangian
        coordinates: List of generalized coordinates
        parameters: Optional parameter values
        
    Returns:
        ModalAnalysisResult
    """
    analyzer = NormalModeAnalyzer()
    return analyzer.analyze(lagrangian, coordinates, parameters)


class CoupledOscillatorSystem:
    """
    Helper class for setting up coupled oscillator systems.
    
    Provides convenient methods for defining common coupled systems:
    - Chain of masses and springs
    - Coupled pendulums
    - 2D lattices
    
    Example:
        >>> system = CoupledOscillatorSystem()
        >>> system.add_mass('m1', 1.0)
        >>> system.add_mass('m2', 1.0)
        >>> system.add_spring('m1', 'wall', k=10.0)
        >>> system.add_spring('m1', 'm2', k=5.0)
        >>> system.add_spring('m2', 'wall', k=10.0)
        >>> L = system.build_lagrangian()
    """
    
    def __init__(self):
        self.masses: Dict[str, float] = {}
        self.springs: List[Tuple[str, str, float]] = []  # (m1, m2, k)
        self._symbol_cache: Dict[str, sp.Symbol] = {}
    
    def get_symbol(self, name: str, **assumptions) -> sp.Symbol:
        """Get or create a symbol."""
        if name not in self._symbol_cache:
            default_assumptions = {'real': True}
            default_assumptions.update(assumptions)
            self._symbol_cache[name] = sp.Symbol(name, **default_assumptions)
        return self._symbol_cache[name]
    
    def add_mass(self, name: str, mass: float) -> None:
        """Add a mass to the system."""
        self.masses[name] = mass
    
    def add_spring(self, mass1: str, mass2: str, k: float) -> None:
        """
        Add a spring between two masses or a mass and wall.
        
        Args:
            mass1: First mass name (or 'wall' for fixed point)
            mass2: Second mass name (or 'wall' for fixed point)
            k: Spring constant
        """
        self.springs.append((mass1, mass2, k))
    
    def build_lagrangian(self) -> sp.Expr:
        """
        Build the Lagrangian for the coupled system.
        
        L = T - V
        T = (1/2) * Σ mᵢ * ẋᵢ²
        V = (1/2) * Σ kᵢⱼ * (xᵢ - xⱼ)²
        
        Returns:
            Symbolic Lagrangian expression
        """
        # Kinetic energy
        T = sp.S.Zero
        for name, mass in self.masses.items():
            x_dot = self.get_symbol(f"{name}_dot")
            T += sp.Rational(1, 2) * mass * x_dot**2
        
        # Potential energy
        V = sp.S.Zero
        for mass1, mass2, k in self.springs:
            if mass1 == 'wall':
                x1 = 0
            else:
                x1 = self.get_symbol(mass1)
            
            if mass2 == 'wall':
                x2 = 0
            else:
                x2 = self.get_symbol(mass2)
            
            V += sp.Rational(1, 2) * k * (x1 - x2)**2
        
        return T - V
    
    def get_coordinates(self) -> List[str]:
        """Get list of coordinate names (mass names)."""
        return list(self.masses.keys())
    
    def analyze(self) -> ModalAnalysisResult:
        """
        Build Lagrangian and perform modal analysis.
        
        Returns:
            ModalAnalysisResult for the coupled system
        """
        L = self.build_lagrangian()
        coordinates = self.get_coordinates()
        
        analyzer = NormalModeAnalyzer()
        return analyzer.analyze(L, coordinates)
