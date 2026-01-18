"""
Tests for JSON Schemas

Run with:
    pytest tests/test_schemas.py -v
"""

import pytest
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent
SCHEMAS_DIR = PROJECT_ROOT / 'schemas'


class TestSchemaFiles:
    """Test schema file validity."""
    
    def test_schemas_directory_exists(self):
        """Test schemas directory exists."""
        assert SCHEMAS_DIR.exists()
        assert SCHEMAS_DIR.is_dir()
    
    def test_config_schema_exists(self):
        """Test config schema exists."""
        schema_path = SCHEMAS_DIR / 'config.schema.json'
        assert schema_path.exists()
    
    def test_config_schema_valid_json(self):
        """Test config schema is valid JSON."""
        schema_path = SCHEMAS_DIR / 'config.schema.json'
        with open(schema_path) as f:
            schema = json.load(f)
        assert '$schema' in schema
        assert 'properties' in schema
    
    def test_simulation_output_schema_exists(self):
        """Test simulation output schema exists."""
        schema_path = SCHEMAS_DIR / 'simulation-output.schema.json'
        assert schema_path.exists()
    
    def test_simulation_output_schema_valid_json(self):
        """Test simulation output schema is valid JSON."""
        schema_path = SCHEMAS_DIR / 'simulation-output.schema.json'
        with open(schema_path) as f:
            schema = json.load(f)
        assert '$schema' in schema
        assert 'properties' in schema


class TestConfigSchemaContent:
    """Test config schema content."""
    
    @pytest.fixture
    def config_schema(self):
        """Load config schema."""
        with open(SCHEMAS_DIR / 'config.schema.json') as f:
            return json.load(f)
    
    def test_has_version_property(self, config_schema):
        """Test schema has version property."""
        assert 'version' in config_schema['properties']
    
    def test_has_project_property(self, config_schema):
        """Test schema has project property."""
        assert 'project' in config_schema['properties']
    
    def test_has_simulation_property(self, config_schema):
        """Test schema has simulation property."""
        assert 'simulation' in config_schema['properties']
    
    def test_has_output_property(self, config_schema):
        """Test schema has output property."""
        assert 'output' in config_schema['properties']
    
    def test_has_codegen_property(self, config_schema):
        """Test schema has codegen property."""
        assert 'codegen' in config_schema['properties']


class TestSchemaValidation:
    """Test schema validation with jsonschema (if available)."""
    
    @pytest.fixture
    def config_schema(self):
        """Load config schema."""
        with open(SCHEMAS_DIR / 'config.schema.json') as f:
            return json.load(f)
    
    def test_valid_config(self, config_schema):
        """Test valid configuration validates."""
        try:
            import jsonschema
        except ImportError:
            pytest.skip("jsonschema not installed")
        
        valid_config = {
            "version": "1.0",
            "project": {
                "name": "test-project"
            },
            "simulation": {
                "t_start": 0,
                "t_end": 10
            }
        }
        
        # Should not raise
        jsonschema.validate(valid_config, config_schema)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
