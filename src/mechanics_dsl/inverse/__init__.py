"""
MechanicsDSL Inverse Problems Module

Parameter estimation, sensitivity analysis, and uncertainty quantification.

Example:
    from mechanics_dsl.inverse import ParameterEstimator
    
    estimator = ParameterEstimator(compiler)
    fitted = estimator.fit(observations, t_obs, ['m', 'k'])
"""

from .estimator import (
    ParameterEstimator,
    FitResult,
)
from .sensitivity import (
    SensitivityAnalyzer,
    SobolIndices,
)
from .uncertainty import (
    UncertaintyQuantifier,
    MCMCResult,
    BootstrapResult,
)

__all__ = [
    'ParameterEstimator',
    'FitResult',
    'SensitivityAnalyzer',
    'SobolIndices',
    'UncertaintyQuantifier',
    'MCMCResult',
    'BootstrapResult',
]
