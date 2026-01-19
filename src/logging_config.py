"""
Structured Logging Configuration.

Provides JSON-structured logging for production observability.
Logs include context like region, temperature, and decision metadata.

Usage:
    from src.logging_config import get_logger
    
    logger = get_logger(__name__)
    logger.info("Model loaded", extra={"model_path": "model.pkl"})
"""

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from src.config import settings


class JSONFormatter(logging.Formatter):
    """
    Formats log records as JSON for structured logging.
    
    Output format is compatible with Azure Monitor, Datadog,
    and other log aggregation services.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as JSON."""
        log_entry: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add application context
        log_entry["app"] = {
            "name": settings.APP_NAME,
            "version": settings.APP_VERSION,
        }
        
        # Add extra fields if present
        if hasattr(record, "__dict__"):
            extra_fields = {
                k: v for k, v in record.__dict__.items()
                if k not in {
                    "name", "msg", "args", "created", "filename", "funcName",
                    "levelname", "levelno", "lineno", "module", "msecs",
                    "pathname", "process", "processName", "relativeCreated",
                    "stack_info", "exc_info", "exc_text", "thread", "threadName",
                    "message", "taskName",
                }
            }
            if extra_fields:
                log_entry["context"] = extra_fields
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry, default=str)


class TextFormatter(logging.Formatter):
    """
    Human-readable text formatter for development.
    
    Provides colorized output when running in a terminal.
    """
    
    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"
    
    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as colored text."""
        color = self.COLORS.get(record.levelname, "")
        reset = self.RESET if color else ""
        
        # Base message
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message = f"{timestamp} | {color}{record.levelname:8}{reset} | {record.name} | {record.getMessage()}"
        
        # Add extra context if present
        extra_fields = {
            k: v for k, v in record.__dict__.items()
            if k not in {
                "name", "msg", "args", "created", "filename", "funcName",
                "levelname", "levelno", "lineno", "module", "msecs",
                "pathname", "process", "processName", "relativeCreated",
                "stack_info", "exc_info", "exc_text", "thread", "threadName",
                "message", "taskName",
            }
        }
        if extra_fields:
            context_str = " | ".join(f"{k}={v}" for k, v in extra_fields.items())
            message += f" | {context_str}"
        
        # Add exception if present
        if record.exc_info:
            message += f"\n{self.formatException(record.exc_info)}"
        
        return message


def setup_logging(
    level: Optional[str] = None,
    format_type: Optional[str] = None,
) -> None:
    """
    Configure the root logger with appropriate handlers.
    
    Args:
        level: Log level (DEBUG, INFO, etc.). Defaults to config.
        format_type: Output format (json, text). Defaults to config.
    """
    level = level or settings.LOG_LEVEL
    format_type = format_type or settings.LOG_FORMAT
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Create handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, level.upper()))
    
    # Set formatter based on config
    if format_type.lower() == "json":
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(TextFormatter())
    
    root_logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name.
    
    Ensures logging is configured on first call.
    
    Args:
        name: Logger name (typically __name__).
    
    Returns:
        Configured logger instance.
    
    Example:
        logger = get_logger(__name__)
        logger.info("Processing started", extra={"region": "Arizona"})
    """
    # Ensure logging is set up
    if not logging.getLogger().handlers:
        setup_logging()
    
    return logging.getLogger(name)


class LoggerAdapter(logging.LoggerAdapter):
    """
    Logger adapter that adds consistent context to all log messages.
    
    Useful for adding request IDs, session IDs, or other context
    that should appear in every log message.
    
    Example:
        base_logger = get_logger(__name__)
        logger = LoggerAdapter(base_logger, {"request_id": "abc-123"})
        logger.info("Processing")  # Will include request_id in output
    """
    
    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple:
        """Add extra context to the log record."""
        extra = kwargs.get("extra", {})
        extra.update(self.extra)
        kwargs["extra"] = extra
        return msg, kwargs


# Log key events that should always be captured
class EventLogger:
    """
    Convenience class for logging key application events.
    
    Provides type-safe methods for common events with
    consistent field names for querying.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        """Initialize with an optional logger."""
        self._logger = logger or get_logger("events")
    
    def model_loaded(
        self,
        model_type: str,
        path: str,
        load_time_ms: Optional[float] = None,
    ) -> None:
        """Log model loading event."""
        self._logger.info(
            "Model loaded",
            extra={
                "event": "model_loaded",
                "model_type": model_type,
                "path": path,
                "load_time_ms": load_time_ms,
            }
        )
    
    def inference_started(
        self,
        region: str,
        n_samples: int,
    ) -> None:
        """Log inference start."""
        self._logger.info(
            "Inference started",
            extra={
                "event": "inference_started",
                "region": region,
                "n_samples": n_samples,
            }
        )
    
    def inference_completed(
        self,
        region: str,
        n_samples: int,
        duration_ms: float,
        max_prediction: float,
    ) -> None:
        """Log inference completion."""
        self._logger.info(
            "Inference completed",
            extra={
                "event": "inference_completed",
                "region": region,
                "n_samples": n_samples,
                "duration_ms": duration_ms,
                "max_prediction": max_prediction,
            }
        )
    
    def risk_breach_detected(
        self,
        region: str,
        temperature: float,
        threshold: float,
    ) -> None:
        """Log thermal risk breach."""
        self._logger.warning(
            "Risk breach detected",
            extra={
                "event": "risk_breach_detected",
                "region": region,
                "temperature": temperature,
                "threshold": threshold,
                "severity": "HIGH",
            }
        )
    
    def scheduling_decision(
        self,
        region: str,
        decision: str,
        reason: str,
        workload_shift_pct: float,
    ) -> None:
        """Log scheduling decision."""
        level = logging.WARNING if decision == "BLOCKED" else logging.INFO
        self._logger.log(
            level,
            f"Scheduling decision: {decision}",
            extra={
                "event": "scheduling_decision",
                "region": region,
                "decision": decision,
                "reason": reason,
                "workload_shift_pct": workload_shift_pct,
            }
        )


# Singleton event logger
event_logger = EventLogger()
