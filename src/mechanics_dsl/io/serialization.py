"""
Serialization utilities for MechanicsDSL

Provides serialization and deserialization of physics systems and solutions.
"""

import json
import pickle
from typing import Any, Dict, Optional

from ..utils import logger, validate_file_path


class SystemSerializer:
    """
    Serializer for physics system configurations.

    Supports JSON and pickle formats for saving/loading system state.
    """

    @staticmethod
    def save_json(data: Dict[str, Any], filename: str) -> bool:
        """
        Save system data to JSON file.

        Args:
            data: Dictionary containing system configuration
            filename: Output file path

        Returns:
            True if successful
        """
        try:
            # Convert non-JSON-serializable types
            clean_data = SystemSerializer._prepare_for_json(data)

            with open(filename, "w") as f:
                json.dump(clean_data, f, indent=2)

            logger.info(f"System saved to {filename}")
            return True
        except Exception as e:
            logger.error(f"Failed to save system: {e}")
            return False

    @staticmethod
    def load_json(filename: str) -> Optional[Dict[str, Any]]:
        """
        Load system data from JSON file.

        Args:
            filename: Input file path

        Returns:
            Dictionary containing system configuration, or None on error
        """
        try:
            validate_file_path(filename, must_exist=True)

            with open(filename, "r") as f:
                data = json.load(f)

            logger.info(f"System loaded from {filename}")
            return data
        except Exception as e:
            logger.error(f"Failed to load system: {e}")
            return None

    @staticmethod
    def save_pickle(data: Dict[str, Any], filename: str) -> bool:
        """Save system data to pickle file (preserves all Python types)."""
        try:
            with open(filename, "wb") as f:
                pickle.dump(data, f)
            logger.info(f"System pickled to {filename}")
            return True
        except Exception as e:
            logger.error(f"Failed to pickle system: {e}")
            return False

    @staticmethod
    def load_pickle(filename: str, allow_pickle: bool = False) -> Optional[Dict[str, Any]]:
        """
        Load system data from a pickle file.

        Pickle deserialization can execute arbitrary code, so this is gated
        behind an explicit ``allow_pickle=True`` flag. Only enable this when
        the file's provenance is fully trusted (e.g. created by you on this
        machine). For anything you didn't write yourself, prefer JSON.
        """
        if not allow_pickle:
            logger.error(
                "load_pickle refused: pickle deserialization can execute arbitrary "
                "code. Pass allow_pickle=True only for fully trusted local files."
            )
            return None

        try:
            validate_file_path(filename, must_exist=True)

            logger.warning(
                f"Loading pickle from {filename}; allow_pickle=True was set. "
                "Only do this for trusted files."
            )
            with open(filename, "rb") as f:
                # nosec B301: caller opted in via allow_pickle=True
                data = pickle.load(f)  # nosec B301

            logger.info(f"System loaded from pickle {filename}")
            return data
        except Exception as e:
            logger.error(f"Failed to load pickle: {e}")
            return None

    @staticmethod
    def _prepare_for_json(data: Any) -> Any:
        """Recursively prepare data for JSON serialization."""
        import numpy as np

        if isinstance(data, dict):
            return {k: SystemSerializer._prepare_for_json(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return [SystemSerializer._prepare_for_json(v) for v in data]
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, (np.integer, np.floating)):
            return float(data)
        elif hasattr(data, "__dict__"):
            return str(data)  # Convert SymPy expressions etc. to strings
        return data


def serialize_solution(solution: Dict[str, Any], filename: str, format: str = "json") -> bool:
    """
    Serialize a simulation solution to file.

    Args:
        solution: Solution dictionary from simulation
        filename: Output file path
        format: 'json' or 'pickle'

    Returns:
        True if successful
    """
    if format == "pickle":
        return SystemSerializer.save_pickle(solution, filename)
    return SystemSerializer.save_json(solution, filename)


def deserialize_solution(
    filename: str, format: str = None, allow_pickle: bool = False
) -> Optional[Dict[str, Any]]:
    """
    Deserialize a simulation solution from file.

    Args:
        filename: Input file path
        format: 'json' or 'pickle' (auto-detected if None)
        allow_pickle: Required for pickle files. Pickle can execute arbitrary
            code, so it is refused by default. See ``SystemSerializer.load_pickle``.

    Returns:
        Solution dictionary or None on error
    """
    if format is None:
        # Auto-detect format
        if filename.endswith(".pkl") or filename.endswith(".pickle"):
            format = "pickle"
        else:
            format = "json"

    if format == "pickle":
        return SystemSerializer.load_pickle(filename, allow_pickle=allow_pickle)
    return SystemSerializer.load_json(filename)
