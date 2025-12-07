"""
Stability Analysis for Classical Mechanics

This module implements:
- Equilibrium point finding
- Linearization around equilibrium
- Eigenvalue analysis for stability classification
- Small oscillation analysis

For a mechanical system near equilibrium q₀, the linearized equations are:
    M * q̈ + K * q = 0
    
where M is the mass matrix and K is the stiffness matrix.
The eigenvalues determine stability:
    - All real negative → stable (damped)
    - All pure imaginary → marginally stable (oscillatory)
    - Any positive real part → unstable
"""
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import sympy as sp
import numpy as np
from numpy.linalg import eig, eigvals

from ...utils import logger


class StabilityType(Enum):
    """Classification of equilibrium stability."""
    STABLE = "stable"                      # Asymptotically stable
    MARGINALLY_STABLE = "marginally_stable"  # Lyapunov stable, not asymptotic
    UNSTABLE = "unstable"                  # At least one growing mode
    SADDLE = "saddle"                      # Mixed stable/unstable directions
    CENTER = "center"                      # Pure oscillatory (conservative)
    UNKNOWN = "unknown"


@dataclass
class EquilibriumPoint:
    """
    Represents an equilibrium point of a mechanical system.
    
    Attributes:
        coordinates: Dictionary of coordinate values at equilibrium
        potential_energy: Value of potential energy at this point
        is_minimum: Whether this is a potential energy minimum
    """
    coordinates: Dict[str, float]
    potential_energy: float = 0.0
    is_minimum: bool = False
    
    def __repr__(self) -> str:
        status = "minimum" if self.is_minimum else "saddle/maximum"
        return f"EquilibriumPoint({self.coordinates}, V={self.potential_energy:.4f}, {status})"


@dataclass
class StabilityResult:
    """
    Result of stability analysis at an equilibrium point.
    
    Attributes:
        equilibrium: The equilibrium point analyzed
        stability_type: Classification of stability
        eigenvalues: Complex eigenvalues of the linearized system
        eigenvectors: Corresponding eigenvectors (mode shapes)
        natural_frequencies: Oscillation frequencies (for stable modes)
        damping_ratios: Damping ratios (for dissipative systems)
    """
    equilibrium: EquilibriumPoint
    stability_type: StabilityType
    eigenvalues: np.ndarray
    eigenvectors: Optional[np.ndarray] = None
    natural_frequencies: Optional[np.ndarray] = None
    damping_ratios: Optional[np.ndarray] = None
    mass_matrix: Optional[np.ndarray] = None
    stiffness_matrix: Optional[np.ndarray] = None
    
    def is_stable(self) -> bool:
        """Check if equilibrium is stable."""
        return self.stability_type in [StabilityType.STABLE, StabilityType.MARGINALLY_STABLE, StabilityType.CENTER]
    
    def get_oscillation_periods(self) -> Optional[np.ndarray]:
        """Get oscillation periods for stable modes."""
        if self.natural_frequencies is None:
            return None
        # Filter out zero frequencies
        nonzero = self.natural_frequencies[self.natural_frequencies > 1e-10]
        return 2 * np.pi / nonzero if len(nonzero) > 0 else None
    
    def __repr__(self) -> str:
        return (f"StabilityResult(type={self.stability_type.value}, "
                f"eigenvalues={self.eigenvalues}, "
                f"frequencies={self.natural_frequencies})")


class StabilityAnalyzer:
    """
    Stability analysis tools for classical mechanical systems.
    
    Provides methods for:
    - Finding equilibrium points (∂V/∂q = 0)
    - Linearizing equations around equilibrium
    - Computing eigenvalues and stability classification
    - Normal mode analysis
    
    Example:
        >>> analyzer = StabilityAnalyzer()
        >>> equilibria = analyzer.find_equilibria(V, coordinates)
        >>> for eq in equilibria:
        ...     result = analyzer.analyze_stability(L, eq, coordinates)
        ...     print(result.stability_type)
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
    
    def find_equilibria(self, potential: sp.Expr, coordinates: List[str],
                        bounds: Optional[Dict[str, Tuple[float, float]]] = None) -> List[EquilibriumPoint]:
        """
        Find equilibrium points where ∂V/∂qᵢ = 0 for all i.
        
        Args:
            potential: Potential energy V(q)
            coordinates: List of generalized coordinates
            bounds: Optional bounds for each coordinate
            
        Returns:
            List of equilibrium points found
        """
        # Create symbols
        q_symbols = [self.get_symbol(q) for q in coordinates]
        
        # Compute gradient of potential
        gradient = [sp.diff(potential, q) for q in q_symbols]
        
        # Solve ∇V = 0
        try:
            solutions = sp.solve(gradient, q_symbols, dict=True)
        except Exception as e:
            logger.warning(f"Could not solve for equilibria analytically: {e}")
            return []
        
        equilibria = []
        for sol in solutions:
            # Check if solution is real
            is_real = True
            coords = {}
            for q_sym, q_name in zip(q_symbols, coordinates):
                if q_sym in sol:
                    val = complex(sol[q_sym].evalf())
                    if abs(val.imag) > 1e-10:
                        is_real = False
                        break
                    coords[q_name] = float(val.real)
                else:
                    coords[q_name] = 0.0  # Assume zero if not in solution
            
            if not is_real:
                continue
            
            # Check bounds if provided
            if bounds:
                in_bounds = True
                for q_name, (low, high) in bounds.items():
                    if q_name in coords:
                        if not (low <= coords[q_name] <= high):
                            in_bounds = False
                            break
                if not in_bounds:
                    continue
            
            # Evaluate potential at equilibrium
            subs_dict = {q_sym: coords[q_name] for q_sym, q_name in zip(q_symbols, coordinates)}
            V_eq = float(potential.subs(subs_dict).evalf())
            
            # Check if it's a minimum (positive definite Hessian)
            is_minimum = self._is_minimum(potential, q_symbols, subs_dict)
            
            equilibria.append(EquilibriumPoint(coords, V_eq, is_minimum))
        
        return equilibria
    
    def _is_minimum(self, potential: sp.Expr, q_symbols: List[sp.Symbol],
                    subs_dict: Dict[sp.Symbol, float]) -> bool:
        """Check if equilibrium is a potential energy minimum."""
        n = len(q_symbols)
        hessian = sp.zeros(n, n)
        
        for i, qi in enumerate(q_symbols):
            for j, qj in enumerate(q_symbols):
                hessian[i, j] = sp.diff(potential, qi, qj)
        
        # Evaluate Hessian at equilibrium
        hessian_eval = np.array(hessian.subs(subs_dict).evalf(), dtype=float)
        
        # Check positive definiteness (all eigenvalues positive)
        eigenvalues = np.linalg.eigvalsh(hessian_eval)
        return np.all(eigenvalues > -1e-10)
    
    def linearize_lagrangian(self, lagrangian: sp.Expr, coordinates: List[str],
                             equilibrium: EquilibriumPoint) -> Tuple[sp.Matrix, sp.Matrix]:
        """
        Linearize Lagrangian about equilibrium to extract M and K matrices.
        
        For small oscillations about equilibrium:
            L ≈ (1/2) * q̇ᵀ * M * q̇ - (1/2) * qᵀ * K * q
        
        Args:
            lagrangian: Full Lagrangian L(q, q̇)
            coordinates: List of generalized coordinates
            equilibrium: Equilibrium point to linearize about
            
        Returns:
            Tuple of (mass_matrix M, stiffness_matrix K)
        """
        n = len(coordinates)
        q_symbols = [self.get_symbol(q) for q in coordinates]
        q_dot_symbols = [self.get_symbol(f"{q}_dot") for q in coordinates]
        
        # Substitution for equilibrium (shift coordinates)
        eq_subs = {self.get_symbol(q): equilibrium.coordinates.get(q, 0.0) 
                   for q in coordinates}
        
        # Mass matrix: Mᵢⱼ = ∂²L/∂q̇ᵢ∂q̇ⱼ evaluated at equilibrium
        M = sp.zeros(n, n)
        for i, qi_dot in enumerate(q_dot_symbols):
            for j, qj_dot in enumerate(q_dot_symbols):
                M[i, j] = sp.diff(lagrangian, qi_dot, qj_dot)
        
        # Evaluate at equilibrium (velocities = 0)
        vel_subs = {qd: 0 for qd in q_dot_symbols}
        M = M.subs(eq_subs).subs(vel_subs)
        
        # Stiffness matrix: Kᵢⱼ = -∂²L/∂qᵢ∂qⱼ evaluated at equilibrium
        # Note: L = T - V, so -∂²L/∂q² = ∂²V/∂q² at q̇ = 0
        K = sp.zeros(n, n)
        for i, qi in enumerate(q_symbols):
            for j, qj in enumerate(q_symbols):
                K[i, j] = -sp.diff(lagrangian, qi, qj)
        
        K = K.subs(eq_subs).subs(vel_subs)
        
        return M, K
    
    def analyze_stability(self, lagrangian: sp.Expr, equilibrium: EquilibriumPoint,
                          coordinates: List[str],
                          damping_matrix: Optional[sp.Matrix] = None) -> StabilityResult:
        """
        Perform complete stability analysis at an equilibrium point.
        
        For the linearized system M*q̈ + C*q̇ + K*q = 0:
        - Compute eigenvalues of the state-space system
        - Classify stability based on eigenvalue locations
        - Extract natural frequencies and damping ratios
        
        Args:
            lagrangian: System Lagrangian L(q, q̇)
            equilibrium: Equilibrium point to analyze
            coordinates: List of generalized coordinates
            damping_matrix: Optional damping matrix C (for dissipative systems)
            
        Returns:
            StabilityResult with complete analysis
        """
        n = len(coordinates)
        
        # Get linearized matrices
        M_sym, K_sym = self.linearize_lagrangian(lagrangian, coordinates, equilibrium)
        
        # Convert to numerical arrays
        try:
            M = np.array(M_sym.evalf(), dtype=float)
            K = np.array(K_sym.evalf(), dtype=float)
        except Exception as e:
            logger.error(f"Could not evaluate matrices numerically: {e}")
            return StabilityResult(
                equilibrium=equilibrium,
                stability_type=StabilityType.UNKNOWN,
                eigenvalues=np.array([])
            )
        
        # Handle damping
        if damping_matrix is not None:
            C = np.array(damping_matrix.evalf(), dtype=float)
        else:
            C = np.zeros((n, n))
        
        # Build state-space system: [q̇, q̈]ᵀ = A * [q, q̇]ᵀ
        # A = [[0, I], [-M⁻¹K, -M⁻¹C]]
        try:
            M_inv = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            logger.error("Mass matrix is singular")
            return StabilityResult(
                equilibrium=equilibrium,
                stability_type=StabilityType.UNKNOWN,
                eigenvalues=np.array([]),
                mass_matrix=M,
                stiffness_matrix=K
            )
        
        # State-space matrix
        A = np.zeros((2*n, 2*n))
        A[:n, n:] = np.eye(n)
        A[n:, :n] = -M_inv @ K
        A[n:, n:] = -M_inv @ C
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = eig(A)
        
        # Classify stability
        stability_type = self._classify_stability(eigenvalues)
        
        # Extract natural frequencies (imaginary parts of eigenvalues)
        natural_frequencies = np.abs(eigenvalues.imag)
        # Keep only unique non-zero frequencies
        natural_frequencies = np.unique(natural_frequencies[natural_frequencies > 1e-10])
        
        # Compute damping ratios for complex eigenvalue pairs
        damping_ratios = self._compute_damping_ratios(eigenvalues)
        
        return StabilityResult(
            equilibrium=equilibrium,
            stability_type=stability_type,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            natural_frequencies=natural_frequencies,
            damping_ratios=damping_ratios,
            mass_matrix=M,
            stiffness_matrix=K
        )
    
    def _classify_stability(self, eigenvalues: np.ndarray) -> StabilityType:
        """Classify stability based on eigenvalue locations."""
        real_parts = eigenvalues.real
        imag_parts = eigenvalues.imag
        
        tol = 1e-10
        
        # Check for positive real parts (unstable)
        has_positive = np.any(real_parts > tol)
        has_negative = np.any(real_parts < -tol)
        all_negative = np.all(real_parts < -tol)
        all_zero_real = np.all(np.abs(real_parts) < tol)
        
        if all_negative:
            return StabilityType.STABLE
        elif has_positive and has_negative:
            return StabilityType.SADDLE
        elif has_positive:
            return StabilityType.UNSTABLE
        elif all_zero_real:
            # Pure imaginary eigenvalues
            if np.any(np.abs(imag_parts) > tol):
                return StabilityType.CENTER
            else:
                return StabilityType.MARGINALLY_STABLE
        else:
            return StabilityType.MARGINALLY_STABLE
    
    def _compute_damping_ratios(self, eigenvalues: np.ndarray) -> np.ndarray:
        """Compute damping ratios for complex eigenvalue pairs."""
        ratios = []
        for lam in eigenvalues:
            if abs(lam.imag) > 1e-10:
                # ζ = -Re(λ) / |λ|
                zeta = -lam.real / abs(lam)
                ratios.append(zeta)
        return np.array(ratios) if ratios else np.array([])
    
    def compute_normal_modes(self, M: np.ndarray, K: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute normal modes from mass and stiffness matrices.
        
        Solves the generalized eigenvalue problem:
            K * φ = ω² * M * φ
        
        Args:
            M: Mass matrix
            K: Stiffness matrix
            
        Returns:
            Tuple of (frequencies, mode_shapes)
        """
        # Solve generalized eigenvalue problem
        try:
            M_inv = np.linalg.inv(M)
            eigenvalues, eigenvectors = eig(M_inv @ K)
        except np.linalg.LinAlgError as e:
            logger.error(f"Could not compute normal modes: {e}")
            return np.array([]), np.array([])
        
        # Eigenvalues are ω², take square root for frequencies
        omega_squared = eigenvalues.real
        # Handle small negative values from numerical errors
        omega_squared = np.maximum(omega_squared, 0)
        frequencies = np.sqrt(omega_squared)
        
        # Sort by frequency
        idx = np.argsort(frequencies)
        frequencies = frequencies[idx]
        mode_shapes = eigenvectors[:, idx]
        
        return frequencies, mode_shapes


def find_equilibria(potential: sp.Expr, coordinates: List[str],
                    bounds: Optional[Dict[str, Tuple[float, float]]] = None) -> List[EquilibriumPoint]:
    """
    Convenience function to find equilibrium points.
    
    Args:
        potential: Potential energy V(q)
        coordinates: List of generalized coordinates
        bounds: Optional bounds for each coordinate
        
    Returns:
        List of equilibrium points found
    """
    analyzer = StabilityAnalyzer()
    return analyzer.find_equilibria(potential, coordinates, bounds)


def analyze_stability(lagrangian: sp.Expr, coordinates: List[str],
                      equilibrium_values: Optional[Dict[str, float]] = None) -> StabilityResult:
    """
    Convenience function for stability analysis.
    
    Args:
        lagrangian: System Lagrangian
        coordinates: List of generalized coordinates
        equilibrium_values: Optional dict of equilibrium coordinate values
        
    Returns:
        StabilityResult for the given equilibrium
    """
    analyzer = StabilityAnalyzer()
    eq_coords = equilibrium_values or {q: 0.0 for q in coordinates}
    equilibrium = EquilibriumPoint(eq_coords)
    return analyzer.analyze_stability(lagrangian, equilibrium, coordinates)
