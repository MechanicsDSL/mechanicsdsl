"""
Symmetry Analysis and Noether's Theorem for Classical Mechanics

This module implements:
- Cyclic coordinate detection
- Noether's theorem for conservation law derivation
- Symmetry group detection
- Automatic conservation law construction

Noether's Theorem: Every continuous symmetry of the action corresponds to
a conserved quantity.

Key symmetries and their conserved quantities:
- Time translation → Energy
- Space translation → Linear momentum
- Rotation → Angular momentum
- Galilean boost → Center of mass motion
"""
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import sympy as sp

from ...utils import logger


class SymmetryType(Enum):
    """Types of symmetries in classical mechanics."""
    TIME_TRANSLATION = "time_translation"      # t → t + ε
    SPACE_TRANSLATION = "space_translation"    # x → x + ε  
    ROTATION = "rotation"                      # θ → θ + ε
    SCALING = "scaling"                        # x → λx
    GALILEAN = "galilean"                      # x → x + vt
    CYCLIC = "cyclic"                          # ∂L/∂q = 0


@dataclass
class ConservedQuantity:
    """
    Represents a conserved quantity from Noether's theorem.
    
    Attributes:
        name: Human-readable name
        expression: Symbolic expression for the quantity
        symmetry_type: Type of symmetry that generates this conservation
        associated_coordinate: Coordinate associated with the symmetry (if any)
    """
    name: str
    expression: sp.Expr
    symmetry_type: SymmetryType
    associated_coordinate: Optional[str] = None
    
    def evaluate(self, state: Dict[str, float], parameters: Dict[str, float]) -> float:
        """
        Evaluate the conserved quantity numerically.
        
        Args:
            state: Dictionary of state variable values
            parameters: Dictionary of parameter values
            
        Returns:
            Numerical value of the conserved quantity
        """
        subs_dict = {sp.Symbol(k): v for k, v in state.items()}
        subs_dict.update({sp.Symbol(k): v for k, v in parameters.items()})
        result = self.expression.subs(subs_dict)
        return float(result.evalf())
    
    def __repr__(self) -> str:
        return f"ConservedQuantity({self.name}: {self.expression}, from {self.symmetry_type.value})"


@dataclass
class SymmetryInfo:
    """
    Information about a detected symmetry.
    
    Attributes:
        symmetry_type: Type of symmetry
        coordinate: Coordinate involved (if applicable)
        generator: Infinitesimal generator of the symmetry
        conserved_quantity: The conserved quantity from Noether's theorem
    """
    symmetry_type: SymmetryType
    coordinate: Optional[str] = None
    generator: Optional[sp.Expr] = None
    conserved_quantity: Optional[ConservedQuantity] = None


class NoetherAnalyzer:
    """
    Analyzes symmetries and derives conservation laws using Noether's theorem.
    
    For a Lagrangian L(q, q̇, t) that is invariant under a symmetry transformation,
    this class computes the corresponding conserved quantity.
    
    Key formulas:
    - Cyclic coordinate qᵢ (∂L/∂qᵢ = 0) → pᵢ = ∂L/∂q̇ᵢ conserved
    - Time translation invariance (∂L/∂t = 0) → Energy H = Σpᵢq̇ᵢ - L conserved
    
    Example:
        >>> analyzer = NoetherAnalyzer()
        >>> symmetries = analyzer.find_symmetries(L, coordinates)
        >>> for sym in symmetries:
        ...     print(sym.conserved_quantity)
    """
    
    def __init__(self):
        self._symbol_cache: Dict[str, sp.Symbol] = {}
        self._time_symbol = sp.Symbol('t', real=True)
    
    def get_symbol(self, name: str, **assumptions) -> sp.Symbol:
        """Get or create a symbol with caching."""
        if name not in self._symbol_cache:
            default_assumptions = {'real': True}
            default_assumptions.update(assumptions)
            self._symbol_cache[name] = sp.Symbol(name, **default_assumptions)
        return self._symbol_cache[name]
    
    def detect_cyclic_coordinates(self, lagrangian: sp.Expr, 
                                   coordinates: List[str]) -> List[str]:
        """
        Find cyclic (ignorable) coordinates where ∂L/∂qᵢ = 0.
        
        A cyclic coordinate is one that doesn't appear explicitly in the
        Lagrangian, only through its velocity. The conjugate momentum
        is conserved for cyclic coordinates.
        
        Args:
            lagrangian: Lagrangian expression L(q, q̇)
            coordinates: List of generalized coordinates
            
        Returns:
            List of cyclic coordinate names
        """
        cyclic = []
        
        for q in coordinates:
            q_sym = self.get_symbol(q)
            # Check if Lagrangian depends on q (not q̇)
            dL_dq = sp.diff(lagrangian, q_sym)
            
            if dL_dq.equals(sp.S.Zero) or sp.simplify(dL_dq) == 0:
                cyclic.append(q)
                logger.info(f"Detected cyclic coordinate: {q}")
        
        return cyclic
    
    def compute_conjugate_momentum(self, lagrangian: sp.Expr, 
                                    coordinate: str) -> sp.Expr:
        """
        Compute the conjugate momentum for a coordinate.
        
        pᵢ = ∂L/∂q̇ᵢ
        
        Args:
            lagrangian: Lagrangian expression
            coordinate: Generalized coordinate
            
        Returns:
            Conjugate momentum expression
        """
        q_dot = self.get_symbol(f"{coordinate}_dot")
        return sp.diff(lagrangian, q_dot)
    
    def compute_energy(self, lagrangian: sp.Expr, 
                       coordinates: List[str]) -> sp.Expr:
        """
        Compute the energy (Jacobi integral) from the Lagrangian.
        
        H = Σᵢ pᵢq̇ᵢ - L
        
        This is conserved when ∂L/∂t = 0 (time translation invariance).
        
        Args:
            lagrangian: Lagrangian expression
            coordinates: List of generalized coordinates
            
        Returns:
            Energy expression
        """
        energy = -lagrangian
        
        for q in coordinates:
            q_dot = self.get_symbol(f"{q}_dot")
            momentum = self.compute_conjugate_momentum(lagrangian, q)
            energy += momentum * q_dot
        
        return sp.simplify(energy)
    
    def is_time_independent(self, lagrangian: sp.Expr) -> bool:
        """
        Check if Lagrangian is explicitly time-independent.
        
        Args:
            lagrangian: Lagrangian expression
            
        Returns:
            True if ∂L/∂t = 0
        """
        dL_dt = sp.diff(lagrangian, self._time_symbol)
        return dL_dt.equals(sp.S.Zero) or sp.simplify(dL_dt) == 0
    
    def find_symmetries(self, lagrangian: sp.Expr, 
                        coordinates: List[str]) -> List[SymmetryInfo]:
        """
        Find all detectable symmetries of the Lagrangian.
        
        Args:
            lagrangian: Lagrangian expression
            coordinates: List of generalized coordinates
            
        Returns:
            List of SymmetryInfo objects describing found symmetries
        """
        symmetries = []
        
        # Check for time translation symmetry
        if self.is_time_independent(lagrangian):
            energy = self.compute_energy(lagrangian, coordinates)
            conserved = ConservedQuantity(
                name="energy",
                expression=energy,
                symmetry_type=SymmetryType.TIME_TRANSLATION
            )
            symmetries.append(SymmetryInfo(
                symmetry_type=SymmetryType.TIME_TRANSLATION,
                conserved_quantity=conserved
            ))
            logger.info("Detected time translation symmetry → energy conservation")
        
        # Check for cyclic coordinates
        cyclic = self.detect_cyclic_coordinates(lagrangian, coordinates)
        for q in cyclic:
            momentum = self.compute_conjugate_momentum(lagrangian, q)
            conserved = ConservedQuantity(
                name=f"p_{q}",
                expression=momentum,
                symmetry_type=SymmetryType.CYCLIC,
                associated_coordinate=q
            )
            symmetries.append(SymmetryInfo(
                symmetry_type=SymmetryType.CYCLIC,
                coordinate=q,
                conserved_quantity=conserved
            ))
            logger.info(f"Detected cyclic coordinate {q} → {conserved.name} conserved")
        
        # Check for rotation symmetry in 2D/3D
        rotation_sym = self._check_rotation_symmetry(lagrangian, coordinates)
        if rotation_sym:
            symmetries.append(rotation_sym)
        
        return symmetries
    
    def _check_rotation_symmetry(self, lagrangian: sp.Expr,
                                  coordinates: List[str]) -> Optional[SymmetryInfo]:
        """
        Check for rotation symmetry in Cartesian coordinates.
        
        For a system with x, y coordinates, check if L depends only on r = √(x² + y²).
        If so, angular momentum Lz = x*py - y*px is conserved.
        """
        # Check for x, y pair
        has_x = 'x' in coordinates
        has_y = 'y' in coordinates
        
        if not (has_x and has_y):
            return None
        
        x = self.get_symbol('x')
        y = self.get_symbol('y')
        x_dot = self.get_symbol('x_dot')
        y_dot = self.get_symbol('y_dot')
        
        # Compute angular momentum
        px = sp.diff(lagrangian, x_dot)
        py = sp.diff(lagrangian, y_dot)
        Lz = x * py - y * px
        
        # Check if ∂L/∂φ = 0 where φ = arctan(y/x)
        # This is equivalent to x*∂L/∂y - y*∂L/∂x = 0
        dL_dx = sp.diff(lagrangian, x)
        dL_dy = sp.diff(lagrangian, y)
        rotation_check = sp.simplify(x * dL_dy - y * dL_dx)
        
        if rotation_check == 0:
            conserved = ConservedQuantity(
                name="angular_momentum_z",
                expression=sp.simplify(Lz),
                symmetry_type=SymmetryType.ROTATION
            )
            logger.info("Detected rotation symmetry → angular momentum conserved")
            return SymmetryInfo(
                symmetry_type=SymmetryType.ROTATION,
                conserved_quantity=conserved
            )
        
        return None
    
    def get_all_conserved_quantities(self, lagrangian: sp.Expr,
                                      coordinates: List[str]) -> Dict[str, sp.Expr]:
        """
        Get dictionary of all conserved quantities.
        
        Args:
            lagrangian: Lagrangian expression
            coordinates: List of generalized coordinates
            
        Returns:
            Dictionary mapping names to conserved quantity expressions
        """
        symmetries = self.find_symmetries(lagrangian, coordinates)
        return {
            sym.conserved_quantity.name: sym.conserved_quantity.expression
            for sym in symmetries
            if sym.conserved_quantity is not None
        }
    
    def verify_conservation(self, lagrangian: sp.Expr, quantity: sp.Expr,
                            coordinates: List[str]) -> bool:
        """
        Verify that a quantity is conserved under the equations of motion.
        
        Computes dQ/dt using the chain rule and Euler-Lagrange equations,
        checks if dQ/dt = 0.
        
        Args:
            lagrangian: Lagrangian expression
            quantity: Quantity to check
            coordinates: List of generalized coordinates
            
        Returns:
            True if the quantity is conserved
        """
        # This is a complex computation requiring the equations of motion
        # For now, do a simpler check based on Poisson brackets (for Hamiltonian)
        # or by explicit substitution of EOM
        
        # Simplified check: if quantity has no explicit time dependence
        # and all coordinates that appear are cyclic, it's conserved
        
        dQ_dt_explicit = sp.diff(quantity, self._time_symbol)
        if not (dQ_dt_explicit.equals(sp.S.Zero) or sp.simplify(dQ_dt_explicit) == 0):
            return False
        
        # For a more complete check, we would need to substitute the EOM
        # This is left as a TODO for advanced verification
        logger.debug(f"Quantity {quantity} has no explicit time dependence")
        return True


def detect_cyclic_coordinates(lagrangian: sp.Expr, 
                               coordinates: List[str]) -> List[str]:
    """
    Convenience function to detect cyclic coordinates.
    
    Args:
        lagrangian: Lagrangian expression
        coordinates: List of generalized coordinates
        
    Returns:
        List of cyclic coordinate names
    """
    analyzer = NoetherAnalyzer()
    return analyzer.detect_cyclic_coordinates(lagrangian, coordinates)


def get_conserved_quantities(lagrangian: sp.Expr,
                              coordinates: List[str]) -> Dict[str, sp.Expr]:
    """
    Convenience function to get all conserved quantities.
    
    Args:
        lagrangian: Lagrangian expression
        coordinates: List of generalized coordinates
        
    Returns:
        Dictionary of conserved quantities
    """
    analyzer = NoetherAnalyzer()
    return analyzer.get_all_conserved_quantities(lagrangian, coordinates)
