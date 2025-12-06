"""
Stability Analysis Tools

Provides linearization and stability analysis for dynamical systems.
"""
from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    import sympy as sp
    HAS_SYMPY = True
except ImportError:
    HAS_SYMPY = False

from ..utils import logger


class StabilityAnalyzer:
    """
    Stability analysis for dynamical systems.
    
    Provides:
    - Fixed point finding
    - Linearization and Jacobian computation
    - Eigenvalue analysis
    - Lyapunov exponent estimation
    """
    
    def __init__(self):
        if not HAS_SYMPY:
            raise ImportError("sympy required for stability analysis")
    
    def find_fixed_points(self, equations: Dict[str, sp.Expr],
                         variables: List[sp.Symbol]) -> List[Dict]:
        """
        Find fixed points where all derivatives are zero.
        
        Args:
            equations: Dict mapping derivative names to expressions
            variables: List of state variable symbols
            
        Returns:
            List of fixed point dictionaries
        """
        # Set all derivatives to zero and solve
        eqs = [eq for eq in equations.values()]
        
        try:
            solutions = sp.solve(eqs, variables, dict=True)
            logger.info(f"Found {len(solutions)} fixed points")
            return solutions
        except Exception as e:
            logger.error(f"Failed to find fixed points: {e}")
            return []
    
    def compute_jacobian(self, equations: Dict[str, sp.Expr],
                        variables: List[sp.Symbol]) -> sp.Matrix:
        """
        Compute the Jacobian matrix of the system.
        
        Args:
            equations: Dict mapping output names to expressions
            variables: List of input variable symbols
            
        Returns:
            SymPy Matrix (Jacobian)
        """
        n = len(variables)
        J = sp.zeros(n, n)
        
        eq_list = list(equations.values())
        
        for i, eq in enumerate(eq_list):
            for j, var in enumerate(variables):
                J[i, j] = sp.diff(eq, var)
        
        return J
    
    def analyze_stability(self, jacobian: sp.Matrix,
                         fixed_point: Dict) -> Dict:
        """
        Analyze stability at a fixed point via eigenvalues.
        
        Args:
            jacobian: Jacobian matrix
            fixed_point: Dict of variable values at fixed point
            
        Returns:
            Stability analysis result
        """
        # Substitute fixed point values
        J_eval = jacobian.subs(fixed_point)
        
        # Compute eigenvalues
        try:
            eigenvalues = list(J_eval.eigenvals().keys())
            eigenvalues_numeric = [complex(sp.N(ev)) for ev in eigenvalues]
        except Exception as e:
            logger.error(f"Eigenvalue computation failed: {e}")
            return {'stable': None, 'error': str(e)}
        
        # Determine stability
        real_parts = [ev.real for ev in eigenvalues_numeric]
        max_real = max(real_parts)
        
        if max_real < -1e-10:
            stability = 'stable'
        elif max_real > 1e-10:
            stability = 'unstable'
        else:
            stability = 'marginally_stable'
        
        return {
            'eigenvalues': eigenvalues_numeric,
            'max_real_part': max_real,
            'stability': stability,
            'jacobian': J_eval,
        }
    
    def estimate_lyapunov_exponent(self, trajectory: np.ndarray,
                                  dt: float,
                                  n_renorm: int = 100) -> float:
        """
        Estimate largest Lyapunov exponent from trajectory data.
        
        Uses Wolf's algorithm with periodic renormalization.
        
        Args:
            trajectory: State trajectory array (n_vars x n_points)
            dt: Time step
            n_renorm: Number of renormalization steps
            
        Returns:
            Estimated Lyapunov exponent
        """
        n_points = trajectory.shape[1]
        step = n_points // n_renorm
        
        if step < 2:
            logger.warning("Not enough data for Lyapunov estimation")
            return 0.0
        
        # Simple estimation based on trajectory divergence
        lyap_sum = 0.0
        count = 0
        
        for i in range(min(n_renorm, n_points - step)):
            idx1 = i * step
            idx2 = idx1 + step
            
            if idx2 >= n_points:
                break
            
            d0 = np.linalg.norm(trajectory[:, idx1+1] - trajectory[:, idx1])
            d1 = np.linalg.norm(trajectory[:, idx2] - trajectory[:, idx2-1])
            
            if d0 > 1e-10 and d1 > 1e-10:
                lyap_sum += np.log(d1 / d0)
                count += 1
        
        if count > 0:
            return lyap_sum / (count * step * dt)
        return 0.0
