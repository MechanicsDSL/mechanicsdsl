"""
System serialization for MechanicsDSL.

This module provides classes for exporting and importing compiled physics
systems to/from various file formats.

Classes:
    SystemSerializer: Serialize and deserialize compiled physics systems.

Example:
    >>> from mechanics_dsl.compiler import PhysicsCompiler, SystemSerializer
    >>> compiler = PhysicsCompiler()
    >>> result = compiler.compile_dsl(source)
    >>> SystemSerializer.export_system(compiler, 'system.json')
"""
import json
import pickle
from typing import Optional, TYPE_CHECKING

from ..utils import logger, validate_file_path

if TYPE_CHECKING:
    from .physics_compiler import PhysicsCompiler

__version__ = "1.5.0"


class SystemSerializer:
    """
    Serialize and deserialize compiled physics systems.
    
    This class provides static methods for exporting compiler state
    to files (JSON or pickle format) and importing it back.
    
    Supported formats:
        - JSON: Human-readable, portable
        - Pickle: Python-specific, preserves types
    
    Example:
        >>> # Export
        >>> SystemSerializer.export_system(compiler, 'system.json')
        True
        
        >>> # Import
        >>> state = SystemSerializer.import_system('system.json')
        >>> if state:
        ...     print(state['system_name'])
    """
    
    @staticmethod
    def export_system(compiler: 'PhysicsCompiler', filename: str, 
                     format: str = 'json') -> bool:
        """
        Export compiled system to file.
        
        Args:
            compiler: PhysicsCompiler instance with compiled system.
            filename: Output filename.
            format: Export format ('json' or 'pickle').
            
        Returns:
            True if export succeeded, False otherwise.
            
        Example:
            >>> compiler.compile_dsl(source)
            >>> SystemSerializer.export_system(compiler, 'pendulum.json')
            True
        """
        try:
            state = {
                'version': __version__,
                'system_name': compiler.system_name,
                'variables': compiler.variables,
                'parameters': compiler.parameters_def,
                'initial_conditions': compiler.initial_conditions,
                'lagrangian': str(compiler.lagrangian) if compiler.lagrangian else None,
                'hamiltonian': str(compiler.hamiltonian) if compiler.hamiltonian else None,
                'coordinates': compiler.get_coordinates(),
                'use_hamiltonian': compiler.use_hamiltonian_formulation,
                'constraints': [str(c) for c in compiler.constraints],
                'transforms': {k: str(v) for k, v in compiler.transforms.items()},
            }
            
            if format == 'json':
                with open(filename, 'w') as f:
                    json.dump(state, f, indent=2)
            elif format == 'pickle':
                with open(filename, 'wb') as f:
                    pickle.dump(state, f)
            else:
                raise ValueError(f"Unknown format: {format}")
            
            logger.info(f"System exported to {filename}")
            return True
            
        except (IOError, OSError, PermissionError, ValueError) as e:
            logger.error(f"Export failed: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected export error: {type(e).__name__}: {e}")
            return False
    
    @staticmethod
    def import_system(filename: str) -> Optional[dict]:
        """
        Import system state from file with validation.
        
        Args:
            filename: Input filename (validated for existence and safety).
            
        Returns:
            System state dictionary, or None if import failed.
            
        Raises:
            TypeError: If filename is not a string.
            ValueError: If filename is invalid.
            FileNotFoundError: If file doesn't exist.
            
        Warning:
            Pickle files can execute arbitrary code when loaded. Only import
            pickle files from trusted sources. Consider using JSON format
            for untrusted data.
            
        Example:
            >>> state = SystemSerializer.import_system('pendulum.json')
            >>> if state:
            ...     print(f"Loaded system: {state['system_name']}")
        """
        validate_file_path(filename, must_exist=True)
        
        try:
            if filename.endswith('.json'):
                with open(filename, 'r', encoding='utf-8') as f:
                    state = json.load(f)
            elif filename.endswith('.pkl') or filename.endswith('.pickle'):
                # Security warning for pickle files
                logger.warning(
                    f"Loading pickle file '{filename}'. WARNING: Pickle files can "
                    "execute arbitrary code. Only load pickle files from trusted sources. "
                    "Consider using JSON format for safer serialization."
                )
                with open(filename, 'rb') as f:
                    state = pickle.load(f)
            else:
                # Try JSON first (safer default)
                with open(filename, 'r', encoding='utf-8') as f:
                    state = json.load(f)
            
            logger.info(f"System imported from {filename}")
            return state
            
        except Exception as e:
            logger.error(f"Import failed: {e}")
            return None


__all__ = ['SystemSerializer']

