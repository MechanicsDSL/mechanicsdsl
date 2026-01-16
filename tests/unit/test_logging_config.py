"""
Tests for Logging Configuration Module
=====================================
"""

import pytest
import logging
import json
from io import StringIO

from mechanics_dsl.logging_config import (
    configure_logging, get_logger, log_operation, log_security_event,
    correlation_context, get_correlation_id, set_correlation_id,
    StructuredFormatter, ColoredFormatter, MetricsHandler,
    log_metrics
)


class TestCorrelationContext:
    """Tests for correlation ID management."""
    
    def test_set_and_get_correlation_id(self):
        """Test setting and getting correlation ID."""
        set_correlation_id("test-id-123")
        assert get_correlation_id() == "test-id-123"
    
    def test_correlation_context(self):
        """Test correlation context manager."""
        with correlation_context("ctx-id-456") as cid:
            assert cid == "ctx-id-456"
            assert get_correlation_id() == "ctx-id-456"
    
    def test_auto_generate_id(self):
        """Test auto-generated correlation ID."""
        with correlation_context() as cid:
            assert cid is not None
            assert len(cid) > 0
    
    def test_nested_context(self):
        """Test nested correlation contexts."""
        with correlation_context("outer") as outer_id:
            assert get_correlation_id() == outer_id
            
            with correlation_context("inner") as inner_id:
                assert get_correlation_id() == inner_id


class TestStructuredFormatter:
    """Tests for JSON structured formatter."""
    
    def test_basic_format(self):
        """Test basic log formatting."""
        formatter = StructuredFormatter()
        
        record = logging.LogRecord(
            name='test', level=logging.INFO, pathname='test.py',
            lineno=10, msg='Test message', args=(), exc_info=None
        )
        
        output = formatter.format(record)
        data = json.loads(output)
        
        assert data['message'] == 'Test message'
        assert data['level'] == 'INFO'
        assert data['logger'] == 'test'
    
    def test_with_correlation_id(self):
        """Test format includes correlation ID."""
        formatter = StructuredFormatter()
        
        with correlation_context("test-cid"):
            record = logging.LogRecord(
                name='test', level=logging.INFO, pathname='test.py',
                lineno=10, msg='Test', args=(), exc_info=None
            )
            
            output = formatter.format(record)
            data = json.loads(output)
            
            assert data['correlation_id'] == 'test-cid'
    
    def test_with_extra_fields(self):
        """Test format includes extra fields."""
        formatter = StructuredFormatter()
        
        record = logging.LogRecord(
            name='test', level=logging.INFO, pathname='test.py',
            lineno=10, msg='Test', args=(), exc_info=None
        )
        record.duration_ms = 123.45
        record.operation = 'test_op'
        
        output = formatter.format(record)
        data = json.loads(output)
        
        assert data['duration_ms'] == 123.45
        assert data['operation'] == 'test_op'


class TestColoredFormatter:
    """Tests for colored console formatter."""
    
    def test_basic_format(self):
        """Test basic colored format."""
        formatter = ColoredFormatter()
        
        record = logging.LogRecord(
            name='test', level=logging.INFO, pathname='test.py',
            lineno=10, msg='Test message', args=(), exc_info=None
        )
        
        output = formatter.format(record)
        
        assert 'Test message' in output
        assert 'test' in output


class TestMetricsHandler:
    """Tests for metrics collection handler."""
    
    def test_request_count(self):
        """Test request counting."""
        handler = MetricsHandler()
        
        for _ in range(5):
            record = logging.LogRecord(
                name='test', level=logging.INFO, pathname='test.py',
                lineno=10, msg='Test', args=(), exc_info=None
            )
            handler.emit(record)
        
        metrics = handler.get_metrics()
        assert metrics['request_count'] == 5
    
    def test_error_count(self):
        """Test error counting."""
        handler = MetricsHandler()
        
        # Info log
        info_record = logging.LogRecord(
            name='test', level=logging.INFO, pathname='test.py',
            lineno=10, msg='Info', args=(), exc_info=None
        )
        handler.emit(info_record)
        
        # Error log
        error_record = logging.LogRecord(
            name='test', level=logging.ERROR, pathname='test.py',
            lineno=10, msg='Error', args=(), exc_info=None
        )
        handler.emit(error_record)
        
        metrics = handler.get_metrics()
        assert metrics['error_count'] == 1
    
    def test_duration_tracking(self):
        """Test duration tracking."""
        handler = MetricsHandler()
        
        record = logging.LogRecord(
            name='test', level=logging.INFO, pathname='test.py',
            lineno=10, msg='Test', args=(), exc_info=None
        )
        record.duration_ms = 100.0
        handler.emit(record)
        
        record2 = logging.LogRecord(
            name='test', level=logging.INFO, pathname='test.py',
            lineno=10, msg='Test2', args=(), exc_info=None
        )
        record2.duration_ms = 50.0
        handler.emit(record2)
        
        metrics = handler.get_metrics()
        assert metrics['total_duration_ms'] == 150.0
    
    def test_operation_tracking(self):
        """Test operation tracking."""
        handler = MetricsHandler()
        
        for _ in range(3):
            record = logging.LogRecord(
                name='test', level=logging.INFO, pathname='test.py',
                lineno=10, msg='Test', args=(), exc_info=None
            )
            record.operation = 'simulate'
            record.duration_ms = 10.0
            handler.emit(record)
        
        metrics = handler.get_metrics()
        assert metrics['operations']['simulate']['count'] == 3
        assert metrics['operations']['simulate']['total_time'] == 30.0


class TestLogOperationDecorator:
    """Tests for log_operation decorator."""
    
    def test_basic_logging(self):
        """Test basic operation logging."""
        @log_operation("test_operation")
        def my_func():
            return 42
        
        result = my_func()
        assert result == 42
    
    def test_exception_logging(self):
        """Test exception is logged."""
        @log_operation("failing_op")
        def fail_func():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            fail_func()


class TestConfigureLogging:
    """Tests for configure_logging function."""
    
    def test_default_config(self):
        """Test default logging configuration."""
        logger = configure_logging()
        
        assert logger.name == 'MechanicsDSL'
        assert len(logger.handlers) > 0
    
    def test_structured_config(self):
        """Test structured logging configuration."""
        logger = configure_logging(structured=True)
        
        # Check handler uses StructuredFormatter
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                assert isinstance(handler.formatter, StructuredFormatter)
    
    def test_get_child_logger(self):
        """Test getting child logger."""
        configure_logging()
        child = get_logger('parser')
        
        assert child.name == 'MechanicsDSL.parser'
