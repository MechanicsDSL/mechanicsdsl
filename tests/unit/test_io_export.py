"""
Unit tests for MechanicsDSL io/export module.

Tests the CSVExporter and JSONExporter classes for data export functionality.
"""

import csv
import json
import os
import tempfile

import numpy as np
import pytest

from mechanics_dsl.io.export import CSVExporter, JSONExporter


class TestCSVExporterExportSolution:
    """Tests for CSVExporter.export_solution method."""

    @pytest.fixture
    def simple_solution(self):
        """Create a simple solution dictionary for testing."""
        return {
            "success": True,
            "t": np.array([0.0, 0.5, 1.0]),
            "y": np.array([[1.0, 0.5, 0.0], [0.0, -0.5, -1.0]]),
            "coordinates": ["x"],
        }

    def test_export_solution_creates_file(self, simple_solution):
        """Test that export_solution creates a file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            temp_path = f.name

        try:
            result = CSVExporter.export_solution(simple_solution, temp_path)
            assert result is True
            assert os.path.exists(temp_path)
        finally:
            os.unlink(temp_path)

    def test_export_solution_with_time(self, simple_solution):
        """Test export with time column included."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            temp_path = f.name

        try:
            CSVExporter.export_solution(simple_solution, temp_path, include_time=True)

            with open(temp_path, "r") as f:
                reader = csv.reader(f)
                header = next(reader)
                assert "t" in header
        finally:
            os.unlink(temp_path)

    def test_export_solution_without_time(self, simple_solution):
        """Test export without time column."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            temp_path = f.name

        try:
            CSVExporter.export_solution(simple_solution, temp_path, include_time=False)

            with open(temp_path, "r") as f:
                reader = csv.reader(f)
                header = next(reader)
                assert "t" not in header
        finally:
            os.unlink(temp_path)

    def test_export_solution_correct_rows(self, simple_solution):
        """Test that export contains correct number of data rows."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            temp_path = f.name

        try:
            CSVExporter.export_solution(simple_solution, temp_path)

            with open(temp_path, "r") as f:
                reader = csv.reader(f)
                rows = list(reader)
                # Header + 3 data rows
                assert len(rows) == 4
        finally:
            os.unlink(temp_path)

    def test_export_solution_coordinate_names(self, simple_solution):
        """Test that coordinate names appear in header."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            temp_path = f.name

        try:
            CSVExporter.export_solution(simple_solution, temp_path)

            with open(temp_path, "r") as f:
                reader = csv.reader(f)
                header = next(reader)
                assert "x" in header
                assert "x_dot" in header
        finally:
            os.unlink(temp_path)

    def test_export_solution_no_coordinates(self):
        """Test export with no coordinate names."""
        solution = {
            "success": True,
            "t": np.array([0.0, 1.0]),
            "y": np.array([[1.0, 2.0], [3.0, 4.0]]),
            "coordinates": [],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            temp_path = f.name

        try:
            result = CSVExporter.export_solution(solution, temp_path)
            assert result is True

            with open(temp_path, "r") as f:
                reader = csv.reader(f)
                header = next(reader)
                # Should have generic y0, y1 names
                assert "y0" in header
                assert "y1" in header
        finally:
            os.unlink(temp_path)

    def test_export_solution_returns_false_on_error(self):
        """Test that export returns False on error."""
        solution = {"t": np.array([0.0]), "y": np.array([[1.0]]), "coordinates": []}

        # Try to write to an invalid path
        result = CSVExporter.export_solution(solution, "/invalid/path/that/does/not/exist.csv")
        assert result is False


class TestCSVExporterExportTable:
    """Tests for CSVExporter.export_table method."""

    def test_export_table_creates_file(self):
        """Test that export_table creates a file."""
        data = {"col1": np.array([1.0, 2.0, 3.0]), "col2": np.array([4.0, 5.0, 6.0])}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            temp_path = f.name

        try:
            result = CSVExporter.export_table(data, temp_path)
            assert result is True
            assert os.path.exists(temp_path)
        finally:
            os.unlink(temp_path)

    def test_export_table_correct_headers(self):
        """Test that table headers are correct."""
        data = {"time": np.array([0.0, 1.0]), "values": np.array([10.0, 20.0])}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            temp_path = f.name

        try:
            CSVExporter.export_table(data, temp_path)

            with open(temp_path, "r") as f:
                reader = csv.reader(f)
                header = next(reader)
                assert "time" in header
                assert "values" in header
        finally:
            os.unlink(temp_path)

    def test_export_table_correct_content(self):
        """Test that table content is correct."""
        data = {"a": np.array([1.5, 2.5]), "b": np.array([3.5, 4.5])}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            temp_path = f.name

        try:
            CSVExporter.export_table(data, temp_path)

            with open(temp_path, "r") as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                row1 = next(reader)
                # Values should be present (order depends on dict order)
                assert len(row1) == 2
        finally:
            os.unlink(temp_path)

    def test_export_table_returns_false_on_error(self):
        """Test that export_table returns False when data causes exception (e.g. empty dict)."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            temp_path = f.name
        try:
            result = CSVExporter.export_table({}, temp_path)
            assert result is False
        finally:
            os.unlink(temp_path)


class TestJSONExporterExportSolution:
    """Tests for JSONExporter.export_solution method."""

    @pytest.fixture
    def simple_solution(self):
        return {
            "success": True,
            "t": np.array([0.0, 0.5, 1.0]),
            "y": np.array([[1.0, 0.5, 0.0], [0.0, -0.5, -1.0]]),
            "coordinates": ["x"],
        }

    def test_export_solution_creates_file(self, simple_solution):
        """Test that export creates a JSON file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            result = JSONExporter.export_solution(simple_solution, temp_path)
            assert result is True
            assert os.path.exists(temp_path)
        finally:
            os.unlink(temp_path)

    def test_export_solution_valid_json(self, simple_solution):
        """Test that exported file is valid JSON."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            JSONExporter.export_solution(simple_solution, temp_path)

            with open(temp_path, "r") as f:
                data = json.load(f)
                assert isinstance(data, dict)
        finally:
            os.unlink(temp_path)

    def test_export_solution_formatted(self, simple_solution):
        """Test non-compact (formatted) export."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            JSONExporter.export_solution(simple_solution, temp_path, compact=False)

            with open(temp_path, "r") as f:
                content = f.read()
                # Formatted JSON should have newlines
                assert "\n" in content
        finally:
            os.unlink(temp_path)

    def test_export_solution_compact(self, simple_solution):
        """Test compact export."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            JSONExporter.export_solution(simple_solution, temp_path, compact=True)

            with open(temp_path, "r") as f:
                content = f.read()
                # Compact JSON should be a single line
                assert content.count("\n") <= 1
        finally:
            os.unlink(temp_path)

    def test_export_solution_arrays_converted(self, simple_solution):
        """Test that numpy arrays are converted to lists."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            JSONExporter.export_solution(simple_solution, temp_path)

            with open(temp_path, "r") as f:
                data = json.load(f)
                assert isinstance(data["t"], list)
                assert isinstance(data["y"], list)
        finally:
            os.unlink(temp_path)

    def test_export_solution_returns_false_on_error(self):
        """Test that export_solution returns False when serialization fails (e.g. circular ref)."""
        solution = {
            "t": np.array([1.0]),
            "y": np.array([[1.0]]),
            "coordinates": ["x"],
        }
        solution["self"] = solution  # Circular reference
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name
        try:
            result = JSONExporter.export_solution(solution, temp_path)
            assert result is False
        finally:
            os.unlink(temp_path)


class TestJSONExporterExportParameters:
    """Tests for JSONExporter.export_parameters method."""

    def test_export_parameters_creates_file(self):
        """Test that parameter export creates a file."""
        params = {"mass": 1.0, "length": 2.0, "gravity": 9.81}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            result = JSONExporter.export_parameters(params, temp_path)
            assert result is True
            assert os.path.exists(temp_path)
        finally:
            os.unlink(temp_path)

    def test_export_parameters_correct_content(self):
        """Test that parameters are correctly exported."""
        params = {"m": 1.5, "k": 10.0}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            JSONExporter.export_parameters(params, temp_path)

            with open(temp_path, "r") as f:
                data = json.load(f)
                assert data["m"] == 1.5
                assert data["k"] == 10.0
        finally:
            os.unlink(temp_path)

    def test_export_parameters_returns_false_on_error(self):
        """Test that export_parameters returns False when json.dump fails (e.g. non-serializable)."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name
        try:
            result = JSONExporter.export_parameters({"k": object()}, temp_path)
            assert result is False
        finally:
            os.unlink(temp_path)


class TestJSONExporterConvertArrays:
    """Tests for JSONExporter._convert_arrays method."""

    def test_convert_numpy_array(self):
        """Test conversion of numpy array."""
        arr = np.array([1.0, 2.0, 3.0])
        result = JSONExporter._convert_arrays(arr)

        assert isinstance(result, list)
        assert result == [1.0, 2.0, 3.0]

    def test_convert_nested_dict(self):
        """Test conversion of nested dictionary with arrays."""
        data = {"array": np.array([1, 2]), "nested": {"inner": np.array([3, 4])}}
        result = JSONExporter._convert_arrays(data)

        assert isinstance(result["array"], list)
        assert isinstance(result["nested"]["inner"], list)

    def test_convert_list_of_arrays(self):
        """Test conversion of list containing arrays."""
        data = [np.array([1, 2]), np.array([3, 4])]
        result = JSONExporter._convert_arrays(data)

        assert isinstance(result, list)
        assert isinstance(result[0], list)
        assert isinstance(result[1], list)

    def test_convert_numpy_scalar(self):
        """Test conversion of numpy scalar types."""
        data = {"int": np.int64(42), "float": np.float64(3.14)}
        result = JSONExporter._convert_arrays(data)

        assert isinstance(result["int"], float)
        assert isinstance(result["float"], float)

    def test_convert_primitive_unchanged(self):
        """Test that primitive types are unchanged."""
        data = {"str": "hello", "int": 42, "float": 3.14, "bool": True}
        result = JSONExporter._convert_arrays(data)

        assert result == data

    def test_convert_2d_array(self):
        """Test conversion of 2D numpy array."""
        arr = np.array([[1, 2], [3, 4]])
        result = JSONExporter._convert_arrays(arr)

        assert isinstance(result, list)
        assert result == [[1, 2], [3, 4]]
