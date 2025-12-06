"""
Data export utilities for MechanicsDSL

Provides exporters for various data formats (CSV, JSON, etc.).
"""
import csv
import json
from typing import Dict, List, Any, Optional
import numpy as np

from ..utils import logger


class CSVExporter:
    """
    Export simulation data to CSV format.
    """
    
    @staticmethod
    def export_solution(solution: Dict[str, Any], filename: str,
                       include_time: bool = True) -> bool:
        """
        Export simulation solution to CSV.
        
        Args:
            solution: Solution dictionary with 't' and 'y' arrays
            filename: Output file path
            include_time: Whether to include time column
            
        Returns:
            True if successful
        """
        try:
            t = solution['t']
            y = solution['y']
            coords = solution.get('coordinates', [])
            
            # Build header
            header = []
            if include_time:
                header.append('t')
            for coord in coords:
                header.extend([coord, f'{coord}_dot'])
            
            # If no coordinate names, use generic
            if not coords:
                for i in range(y.shape[0]):
                    header.append(f'y{i}')
            
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                
                for i in range(len(t)):
                    row = []
                    if include_time:
                        row.append(t[i])
                    row.extend(y[:, i].tolist())
                    writer.writerow(row)
            
            logger.info(f"Solution exported to {filename}")
            return True
        except Exception as e:
            logger.error(f"CSV export failed: {e}")
            return False
    
    @staticmethod
    def export_table(data: Dict[str, np.ndarray], filename: str) -> bool:
        """
        Export a dictionary of arrays as columns.
        
        Args:
            data: Dictionary mapping column names to arrays
            filename: Output file path
            
        Returns:
            True if successful
        """
        try:
            headers = list(data.keys())
            columns = [data[h] for h in headers]
            n_rows = len(columns[0])
            
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                
                for i in range(n_rows):
                    row = [col[i] for col in columns]
                    writer.writerow(row)
            
            logger.info(f"Table exported to {filename}")
            return True
        except Exception as e:
            logger.error(f"Table export failed: {e}")
            return False


class JSONExporter:
    """
    Export simulation data to JSON format.
    """
    
    @staticmethod
    def export_solution(solution: Dict[str, Any], filename: str,
                       compact: bool = False) -> bool:
        """
        Export simulation solution to JSON.
        
        Args:
            solution: Solution dictionary
            filename: Output file path
            compact: If True, use minimal formatting
            
        Returns:
            True if successful
        """
        try:
            # Convert numpy arrays to lists
            export_data = JSONExporter._convert_arrays(solution)
            
            indent = None if compact else 2
            
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=indent)
            
            logger.info(f"Solution exported to {filename}")
            return True
        except Exception as e:
            logger.error(f"JSON export failed: {e}")
            return False
    
    @staticmethod
    def export_parameters(parameters: Dict[str, Any], filename: str) -> bool:
        """Export system parameters to JSON."""
        try:
            with open(filename, 'w') as f:
                json.dump(parameters, f, indent=2)
            logger.info(f"Parameters exported to {filename}")
            return True
        except Exception as e:
            logger.error(f"Parameter export failed: {e}")
            return False
    
    @staticmethod
    def _convert_arrays(data: Any) -> Any:
        """Recursively convert numpy arrays to lists."""
        if isinstance(data, dict):
            return {k: JSONExporter._convert_arrays(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return [JSONExporter._convert_arrays(v) for v in data]
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, (np.integer, np.floating)):
            return float(data)
        return data
