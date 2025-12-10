"""
Unit tests for MechanicsDSL io/serialization module.

Tests the SystemSerializer class and serialization utility functions.
"""

import pytest
import numpy as np
import json
import pickle
import os
import tempfile

from mechanics_dsl.io.serialization import (
    SystemSerializer, serialize_solution, deserialize_solution
)


class TestSystemSerializerSaveJson:
    """Tests for SystemSerializer.save_json method."""
    
    def test_save_json_creates_file(self):
        """Test that save_json creates a file."""
        data = {'name': 'pendulum', 'params': {'m': 1.0, 'l': 1.0}}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            result = SystemSerializer.save_json(data, temp_path)
            assert result is True
            assert os.path.exists(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_save_json_valid_content(self):
        """Test that saved JSON is valid."""
        data = {'key': 'value', 'number': 42}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            SystemSerializer.save_json(data, temp_path)
            
            with open(temp_path, 'r') as f:
                loaded = json.load(f)
                assert loaded['key'] == 'value'
                assert loaded['number'] == 42
        finally:
            os.unlink(temp_path)
    
    def test_save_json_with_numpy_arrays(self):
        """Test saving data with numpy arrays."""
        data = {
            'array': np.array([1.0, 2.0, 3.0]),
            'matrix': np.array([[1, 2], [3, 4]])
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            result = SystemSerializer.save_json(data, temp_path)
            assert result is True
            
            with open(temp_path, 'r') as f:
                loaded = json.load(f)
                assert 'array' in loaded
                assert 'matrix' in loaded
        finally:
            os.unlink(temp_path)
    
    def test_save_json_returns_false_on_error(self):
        """Test that save returns False on error."""
        data = {'test': 'data'}
        result = SystemSerializer.save_json(data, '/invalid/path/file.json')
        assert result is False


class TestSystemSerializerLoadJson:
    """Tests for SystemSerializer.load_json method."""
    
    def test_load_json_returns_dict(self):
        """Test that load_json returns a dictionary."""
        data = {'key': 'value'}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            temp_path = f.name
        
        try:
            loaded = SystemSerializer.load_json(temp_path)
            assert isinstance(loaded, dict)
            assert loaded['key'] == 'value'
        finally:
            os.unlink(temp_path)
    
    def test_load_json_nonexistent_file(self):
        """Test loading a nonexistent file returns None."""
        result = SystemSerializer.load_json('/nonexistent/file.json')
        assert result is None
    
    def test_load_json_roundtrip(self):
        """Test save and load roundtrip."""
        original = {
            'system': 'oscillator',
            'params': {'k': 10.0, 'm': 2.0},
            'initial': [1.0, 0.0]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            SystemSerializer.save_json(original, temp_path)
            loaded = SystemSerializer.load_json(temp_path)
            
            assert loaded['system'] == original['system']
            assert loaded['params'] == original['params']
            assert loaded['initial'] == original['initial']
        finally:
            os.unlink(temp_path)


class TestSystemSerializerSavePickle:
    """Tests for SystemSerializer.save_pickle method."""
    
    def test_save_pickle_creates_file(self):
        """Test that save_pickle creates a file."""
        data = {'name': 'system', 'value': 42}
        
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as f:
            temp_path = f.name
        
        try:
            result = SystemSerializer.save_pickle(data, temp_path)
            assert result is True
            assert os.path.exists(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_save_pickle_preserves_numpy(self):
        """Test that pickle preserves numpy arrays exactly."""
        data = {
            'array': np.array([1.0, 2.0, 3.0]),
            'matrix': np.eye(3)
        }
        
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as f:
            temp_path = f.name
        
        try:
            SystemSerializer.save_pickle(data, temp_path)
            
            with open(temp_path, 'rb') as f:
                loaded = pickle.load(f)
                np.testing.assert_array_equal(loaded['array'], data['array'])
                np.testing.assert_array_equal(loaded['matrix'], data['matrix'])
        finally:
            os.unlink(temp_path)
    
    def test_save_pickle_returns_false_on_error(self):
        """Test that save returns False on error."""
        data = {'test': 'data'}
        result = SystemSerializer.save_pickle(data, '/invalid/path/file.pkl')
        assert result is False


class TestSystemSerializerLoadPickle:
    """Tests for SystemSerializer.load_pickle method."""
    
    def test_load_pickle_returns_dict(self):
        """Test that load_pickle returns correct data."""
        data = {'key': 'value', 'number': 42}
        
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as f:
            pickle.dump(data, f)
            temp_path = f.name
        
        try:
            loaded = SystemSerializer.load_pickle(temp_path)
            assert isinstance(loaded, dict)
            assert loaded == data
        finally:
            os.unlink(temp_path)
    
    def test_load_pickle_nonexistent_file(self):
        """Test loading nonexistent file returns None."""
        result = SystemSerializer.load_pickle('/nonexistent/file.pkl')
        assert result is None
    
    def test_load_pickle_roundtrip(self):
        """Test save and load roundtrip."""
        original = {
            'array': np.array([1, 2, 3]),
            'nested': {'inner': np.linspace(0, 1, 10)}
        }
        
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as f:
            temp_path = f.name
        
        try:
            SystemSerializer.save_pickle(original, temp_path)
            loaded = SystemSerializer.load_pickle(temp_path)
            
            np.testing.assert_array_equal(loaded['array'], original['array'])
            np.testing.assert_array_equal(loaded['nested']['inner'], original['nested']['inner'])
        finally:
            os.unlink(temp_path)


class TestSystemSerializerPrepareForJson:
    """Tests for SystemSerializer._prepare_for_json method."""
    
    def test_prepare_numpy_array(self):
        """Test preparation of numpy arrays."""
        data = {'arr': np.array([1, 2, 3])}
        result = SystemSerializer._prepare_for_json(data)
        
        assert isinstance(result['arr'], list)
    
    def test_prepare_numpy_scalars(self):
        """Test preparation of numpy scalar types."""
        data = {
            'int': np.int64(42),
            'float': np.float64(3.14)
        }
        result = SystemSerializer._prepare_for_json(data)
        
        assert isinstance(result['int'], float)
        assert isinstance(result['float'], float)
    
    def test_prepare_nested_structure(self):
        """Test preparation of nested data structures."""
        data = {
            'level1': {
                'level2': {
                    'array': np.array([1, 2])
                }
            }
        }
        result = SystemSerializer._prepare_for_json(data)
        
        assert isinstance(result['level1']['level2']['array'], list)
    
    def test_prepare_list_with_arrays(self):
        """Test preparation of lists containing arrays."""
        data = [np.array([1, 2]), np.array([3, 4])]
        result = SystemSerializer._prepare_for_json(data)
        
        assert isinstance(result, list)
        assert isinstance(result[0], list)
    
    def test_prepare_primitives_unchanged(self):
        """Test that primitive types are preserved."""
        data = {'str': 'hello', 'int': 42, 'float': 3.14}
        result = SystemSerializer._prepare_for_json(data)
        
        assert result == data


class TestSerializeSolution:
    """Tests for serialize_solution function."""
    
    @pytest.fixture
    def solution(self):
        return {
            'success': True,
            't': np.linspace(0, 10, 100),
            'y': np.random.randn(2, 100),
            'coordinates': ['x']
        }
    
    def test_serialize_solution_json(self, solution):
        """Test serialization to JSON format."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            result = serialize_solution(solution, temp_path, format='json')
            assert result is True
            assert os.path.exists(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_serialize_solution_pickle(self, solution):
        """Test serialization to pickle format."""
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as f:
            temp_path = f.name
        
        try:
            result = serialize_solution(solution, temp_path, format='pickle')
            assert result is True
            assert os.path.exists(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_serialize_solution_default_json(self, solution):
        """Test that default format is JSON."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            result = serialize_solution(solution, temp_path)  # No format specified
            assert result is True
            
            # Should be valid JSON
            with open(temp_path, 'r') as f:
                json.load(f)
        finally:
            os.unlink(temp_path)


class TestDeserializeSolution:
    """Tests for deserialize_solution function."""
    
    @pytest.fixture
    def solution(self):
        return {
            'success': True,
            't': np.linspace(0, 5, 50).tolist(),
            'y': [[1, 2, 3], [4, 5, 6]],
            'coordinates': ['x']
        }
    
    def test_deserialize_json(self, solution):
        """Test deserialization from JSON."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(solution, f)
            temp_path = f.name
        
        try:
            loaded = deserialize_solution(temp_path, format='json')
            assert loaded is not None
            assert loaded['success'] == True
        finally:
            os.unlink(temp_path)
    
    def test_deserialize_pickle(self):
        """Test deserialization from pickle."""
        solution = {
            'success': True,
            't': np.linspace(0, 5, 50),
            'y': np.random.randn(2, 50)
        }
        
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as f:
            pickle.dump(solution, f)
            temp_path = f.name
        
        try:
            loaded = deserialize_solution(temp_path, format='pickle')
            assert loaded is not None
            np.testing.assert_array_equal(loaded['t'], solution['t'])
        finally:
            os.unlink(temp_path)
    
    def test_deserialize_auto_detect_json(self, solution):
        """Test auto-detection of JSON format."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(solution, f)
            temp_path = f.name
        
        try:
            # No format specified, should auto-detect from extension
            loaded = deserialize_solution(temp_path)
            assert loaded is not None
        finally:
            os.unlink(temp_path)
    
    def test_deserialize_auto_detect_pickle(self):
        """Test auto-detection of pickle format from .pkl extension."""
        solution = {'test': 'data'}
        
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as f:
            pickle.dump(solution, f)
            temp_path = f.name
        
        try:
            loaded = deserialize_solution(temp_path)  # Auto-detect
            assert loaded == solution
        finally:
            os.unlink(temp_path)
    
    def test_deserialize_nonexistent_file(self):
        """Test deserializing nonexistent file returns None."""
        result = deserialize_solution('/nonexistent/file.json')
        assert result is None
