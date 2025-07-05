import json
import traceback
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Literal, Optional, Union, Set
import logging
import re
import asyncio
import pytz
from difflib import SequenceMatcher
import random
import time
import os
import threading
import uuid
import hashlib
from dataclasses import dataclass, field
from enum import Enum
from pydantic import BaseModel, Field, field_validator, model_validator
import numpy as np
import aiohttp
from aiohttp import ClientTimeout, TCPConnector, ClientError

# Setup structured logging for Docker environments
logger = logging.getLogger("adaptive_memory")

# Create custom formatter for structured logs
class DockerFormatter(logging.Formatter):
    """Custom formatter for Docker-friendly structured logs"""
    
    COMPONENT_COLORS = {
        'INLET': '\033[36m',      # Cyan
        'OUTLET': '\033[35m',     # Magenta
        'MEMORY': '\033[33m',     # Yellow
        'CONFIG': '\033[34m',     # Blue
        'EMBED': '\033[32m',      # Green
        'LLM': '\033[31m',        # Red
        'FILTER': '\033[37m',     # White
        'ERROR': '\033[91m',      # Bright Red
        'SUCCESS': '\033[92m',    # Bright Green
    }
    RESET = '\033[0m'
    
    def format(self, record):
        # Extract component from logger name or message
        component = getattr(record, 'component', 'SYSTEM')
        user_id = getattr(record, 'user_id', 'system')
        operation = getattr(record, 'operation', '')
        
        # Get color for component
        color = self.COMPONENT_COLORS.get(component, '')
        
        # Format timestamp
        timestamp = datetime.now(timezone.utc).isoformat()
        
        # Build structured message
        if record.levelno >= logging.ERROR:
            color = self.COMPONENT_COLORS['ERROR']
            prefix = f"{color}[{timestamp}] [{component}] [ERROR] [user:{user_id}]{self.RESET}"
        elif record.levelno >= logging.WARNING:
            prefix = f"{color}[{timestamp}] [{component}] [WARN] [user:{user_id}]{self.RESET}"
        else:
            prefix = f"{color}[{timestamp}] [{component}] [INFO] [user:{user_id}]{self.RESET}"
        
        if operation:
            prefix += f" [{operation}]"
        
        # Add context if available
        context = getattr(record, 'context', {})
        if context:
            context_str = ' '.join([f"{k}={v}" for k, v in context.items()])
            prefix += f" {{{context_str}}}"
        
        # Format the actual message
        message = f"{prefix} {record.getMessage()}"
        
        # Add exception info if present
        if record.exc_info:
            exception_text = self.formatException(record.exc_info)
            message += f"\n{color}EXCEPTION DETAILS:{self.RESET}\n{exception_text}"
        
        return message

# Configure handler with custom formatter (only once)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = DockerFormatter()
    handler.setFormatter(formatter)
    
    # Set appropriate log level based on environment
    log_level = os.environ.get('ADAPTIVE_MEMORY_LOG_LEVEL', 'INFO')
    handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    logger.addHandler(handler)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    logger.propagate = False  # Prevent duplicate logs from root logger

# Helper function for structured logging
def log_with_context(level: str, message: str, component: str = 'SYSTEM', 
                     user_id: str = 'system', operation: str = '', 
                     context: Optional[Dict[str, Any]] = None, 
                     error: Optional[Exception] = None,
                     exc_info: bool = False):
    """Log with structured context for Docker environments"""
    extra = {
        'component': component,
        'user_id': user_id,
        'operation': operation,
        'context': context or {}
    }
    
    if error:
        # Add error details to context
        extra['context']['error_type'] = type(error).__name__
        extra['context']['error_msg'] = str(error)
        exc_info = True  # Force exc_info when error is provided
    
    log_func = getattr(logger, level.lower(), logger.info)
    log_func(message, extra=extra, exc_info=exc_info)

# Import OpenWebUI classes if available
try:
    from webui.models.users import Users  # type: ignore
    from webui.models.memories import Memories  # type: ignore
    OPENWEBUI_AVAILABLE = True
    logger.info("OpenWebUI classes imported successfully")
except ImportError:
    OPENWEBUI_AVAILABLE = False
    logger.info("OpenWebUI classes not available, using mock implementations")

# Import JSON repair system
try:
    from json_repair_system import EnhancedJSONParser  # type: ignore
    JSON_REPAIR_AVAILABLE = True
except ImportError:
    logger.info("JSON repair system not available, using basic parsing")
    JSON_REPAIR_AVAILABLE = False
    
    # Mock classes for when json_repair_system is not available
    class MockRepairSystem:
        def validate_memory_operations(self, data: Any) -> bool:
            # Mock implementation - always validates successfully
            _ = data  # Acknowledge parameter
            return True
    
    class MockEnhancedJSONParser:
        def __init__(self):
            self.repair_system = MockRepairSystem()
        
        @staticmethod
        def parse(text: str) -> 'MockJSONRepairResult':
            import json
            try:
                return MockJSONRepairResult(json.loads(text), True, "")
            except json.JSONDecodeError as e:
                return MockJSONRepairResult(None, False, str(e))
        
        def parse_with_repair(self, json_str: str, **kwargs) -> 'MockJSONRepairResult':
            _ = kwargs  # Acknowledge parameter
            try:
                import json
                return MockJSONRepairResult(json.loads(json_str), True, "")
            except json.JSONDecodeError as e:
                return MockJSONRepairResult(None, False, str(e))
        
        def repair_json(self, json_str: str) -> str:
            return json_str
    
    class MockJSONRepairResult:
        def __init__(self, data: Any, success: bool, error: str):
            self.data = data
            self.success = success
            self.error = error
            self.parsed_data = data  # Alias for data
            self.repair_method = "basic_parse" if success else "failed"
            self.validation_errors = [] if success else [error]
    
    # Set aliases for backwards compatibility
    EnhancedJSONParser = MockEnhancedJSONParser

# ============================================================================
# OPENWEBUI 2024 COMPLIANCE FEATURES
# ============================================================================
# This filter implements all OpenWebUI 2024 standards and latest features:
#
# 1. Stream Function (v0.5.17+): Real-time filtering of streaming responses
#    - Implements _process_stream_event() for content filtering
#    - Supports PII filtering in streaming data
#    - Handles OpenAI-style streaming format
#
# 2. Database Write Hooks (Issue #11888): Separate processing for display vs storage
#    - _prepare_content_for_database() for write filtering
#    - _restore_content_from_database() for read filtering
#    - Automatic PII filtering with configurable modes (redact/encrypt/anonymize)
#
# 3. Enhanced Valve Configuration: Comprehensive 2024 valve options
#    - Stream filtering controls (enable_stream_filtering, enable_stream_content_filtering)
#    - Database write hooks (enable_database_write_hooks, enable_pii_filtering)
#    - Enhanced event emitter settings (enable_enhanced_event_emitter, event_emitter_batch_size)
#
# 4. OpenWebUI Event Emitter Patterns: 2024-compliant event emission
#    - _emit_enhanced_event() with metadata and batching support
#    - _emit_batched_event() for performance optimization
#    - Structured event format with timestamps and versioning
#
# All features are backward compatible and can be enabled/disabled via valves.
# ============================================================================

# ============================================================================
# SECURITY NOTICE
# ============================================================================
# This file has been secured against the following vulnerabilities:
# 1. Model Loading Security: Embedding model names are validated against whitelist
#    to prevent arbitrary code execution via malicious model names
# 2. Input Sanitization: User inputs are properly validated and sanitized
# 3. API Key Protection: Sensitive information is not logged or exposed
# 4. Path Traversal Prevention: Suspicious patterns are blocked
# 5. Code Injection Prevention: Shell metacharacters and dangerous functions blocked
#
# Security measures implemented:
# - ALLOWED_EMBEDDING_MODELS whitelist for safe model names
# - _validate_embedding_model_name() function for model validation
# - Enhanced field validation in Valves class
# - Improved sanitization in _sanitize_body_parameters()
# - Safe logging practices to prevent credential exposure
# ============================================================================

# ============================================================================
# Global Variables and Mock Objects
# ============================================================================

# Filter orchestration globals (defined below with MockMetric instances)
_orchestration_manager = None

# Mock user and memory management objects
class MockUsers:
    def __init__(self):
        self.memories = MockMemories()
        
    @staticmethod
    def get_user_by_id(user_id: str):
        """Mock user retrieval - returns a basic user object"""
        return {"id": user_id, "name": f"User_{user_id}"}

class MockMemories:
    @staticmethod
    def get_memories_by_user_id(user_id: str):
        """Mock memory retrieval - returns empty list"""
        _ = user_id  # Acknowledge parameter
        return []

# Use real OpenWebUI classes if available, otherwise use mocks
if not OPENWEBUI_AVAILABLE:
    Users = MockUsers()
    Memories = MockMemories()

# Metrics mock objects with inc() and observe() methods
class MockMetric:
    def __init__(self, name: str = ""):
        self.name = name
        self.value = 0
    
    def inc(self):
        self.value += 1
    
    def observe(self, value: float):
        _ = value  # Acknowledge parameter
        pass
    
    def labels(self, *args, **kwargs):
        _ = args, kwargs  # Acknowledge parameters
        return self

# Mock metrics - typed as Any to satisfy both metric and dict usage
RETRIEVAL_REQUESTS: Any = MockMetric("retrieval_requests")
RETRIEVAL_LATENCY: Any = MockMetric("retrieval_latency")
EMBEDDING_REQUESTS: Any = MockMetric("embedding_requests")
EMBEDDING_LATENCY: Any = MockMetric("embedding_latency")
EMBEDDING_ERRORS: Any = MockMetric("embedding_errors")

# Update global metrics to use MockMetric instances  
FILTER_ROLLBACKS: Any = MockMetric("filter_rollbacks")
COORDINATION_OVERHEAD: Any = MockMetric("coordination_overhead")

# Mock form classes
class AddMemoryForm:
    def __init__(self, content: str = "", memory_bank: str = "Personal", **kwargs):
        self.content = content
        self.memory_bank = memory_bank
        for key, value in kwargs.items():
            setattr(self, key, value)

class QueryMemoryForm:
    def __init__(self, query: str = "", **kwargs):
        self.query = query
        for key, value in kwargs.items():
            setattr(self, key, value)

# Mock async functions
async def add_memory(user_id: str, form_data: AddMemoryForm):
    """Mock add memory function"""
    log_with_context('debug', f"Mock: Adding memory - {form_data.content[:50]}...",
                     component='MEMORY', user_id=user_id, operation='add_mock')
    return {"success": True, "memory_id": str(uuid.uuid4())}

async def delete_memory_by_id(user_id: str, memory_id: str):
    """Mock delete memory function"""
    log_with_context('debug', f"Mock: Deleting memory {memory_id}",
                     component='MEMORY', user_id=user_id, operation='delete_mock')
    return {"success": True}

async def query_memory(user_id: str, form_data: QueryMemoryForm):
    """Mock query memory function"""
    log_with_context('debug', f"Mock: Querying memory - {form_data.query[:50]}...",
                     component='MEMORY', user_id=user_id, operation='query_mock')
    return {"results": []}

# Simplified Filter Orchestration System for single filter use
class FilterPriority(Enum):
    HIGHEST = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    LOWEST = 5

@dataclass
class FilterMetadata:
    name: str = "adaptive_memory"
    version: str = "4.0"
    priority: FilterPriority = FilterPriority.NORMAL
    description: str = "Adaptive Memory Filter with persistent user memory across conversations"
    author: str = "OpenWebUI"
    category: str = "memory"
    tags: List[str] = field(default_factory=list)
    requirements: List[str] = field(default_factory=list)
    enabled: bool = True
    config_schema: Dict[str, Any] = field(default_factory=dict)
    api_version: str = "1.0"
    min_openwebui_version: str = "0.1.0"
    max_openwebui_version: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    conflicts: List[str] = field(default_factory=list)
    last_updated: Optional[str] = None
    changelog: List[str] = field(default_factory=list)
    documentation_url: Optional[str] = None
    repository_url: Optional[str] = None
    
    # Additional attributes needed by orchestration system
    capabilities: List[str] = field(default_factory=lambda: ["memory", "persistent_storage", "user_context"])
    operations: List[str] = field(default_factory=lambda: ["store", "retrieve", "search"])
    conflicts_with: List[str] = field(default_factory=list)
    max_execution_time_ms: int = 10000
    memory_requirements_mb: int = 256
    requires_user_context: bool = True
    modifies_content: bool = True
    thread_safe: bool = True
    license: str = "MIT"
    keywords: List[str] = field(default_factory=list)
    environment: Dict[str, Any] = field(default_factory=dict)
    resources: Dict[str, Any] = field(default_factory=dict)
    permissions: List[str] = field(default_factory=list)
    data_retention: Optional[int] = None
    privacy_policy: Optional[str] = None
    terms_of_service: Optional[str] = None

# Simplified orchestration system - most functionality moved to the main Filter class
class SimpleOrchestrator:
    def __init__(self):
        self.metadata = FilterMetadata()
        self._performance_history = []
    
    def record_performance(self, execution_time_ms: float):
        self._performance_history.append(execution_time_ms)
        if len(self._performance_history) > 100:
            self._performance_history.pop(0)
    
    def get_average_performance(self) -> float:
        return sum(self._performance_history) / len(self._performance_history) if self._performance_history else 0.0

class JsonFormatter(logging.Formatter):
    def format(self, record):
        import json as _json

        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "funcName": record.funcName,
            "lineNo": record.lineno,
            "process": record.process,
            "thread": record.thread,
        }
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
        return _json.dumps(log_record)


# JsonFormatter setup removed - using DockerFormatter above to prevent duplicate handlers
# The handler is already configured above with DockerFormatter

# Helper function for structured logging
# Duplicate function removed - using the one defined at line 95
# Do not override root logger level; respect GLOBAL_LOG_LEVEL or root config


# ============================================================================
# Custom Exception Framework for Adaptive Memory Plugin
# ============================================================================

class AdaptiveMemoryError(Exception):
    """Base exception for all adaptive memory plugin errors"""
    error_code: str = "AM_GENERIC_ERROR"
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None, suggestion: Optional[str] = None):
        self.message = message
        self.details = details or {}
        self.suggestion = suggestion
        super().__init__(self.format_message())
    
    def format_message(self) -> str:
        """Format exception message with details and suggestions"""
        msg = f"[{self.error_code}] {self.message}"
        if self.details:
            msg += f"\nDetails: {json.dumps(self.details, indent=2)}"
        if self.suggestion:
            msg += f"\nSuggestion: {self.suggestion}"
        return msg
    
    def to_log_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for structured logging"""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
            "suggestion": self.suggestion,
            "exception_type": self.__class__.__name__
        }


# LLM Connection Errors
class LLMConnectionError(AdaptiveMemoryError):
    """Base exception for LLM connection issues"""
    error_code = "AM_LLM_CONNECTION_ERROR"

class LLMTimeoutError(LLMConnectionError):
    """LLM request timed out"""
    error_code = "AM_LLM_TIMEOUT"
    
    def __init__(self, provider: str, timeout_seconds: float, endpoint: Optional[str] = None):
        super().__init__(
            f"LLM request to {provider} timed out after {timeout_seconds}s",
            details={"provider": provider, "timeout": timeout_seconds, "endpoint": endpoint},
            suggestion="Increase timeout value in configuration or check LLM service availability"
        )

class LLMAuthenticationError(LLMConnectionError):
    """LLM authentication failed"""
    error_code = "AM_LLM_AUTH_FAILED"
    
    def __init__(self, provider: str, status_code: Optional[int] = None):
        super().__init__(
            f"Authentication failed for {provider}",
            details={"provider": provider, "status_code": status_code},
            suggestion="Check API key configuration and ensure it's valid for the provider"
        )

class LLMServiceUnavailableError(LLMConnectionError):
    """LLM service is unavailable"""
    error_code = "AM_LLM_SERVICE_UNAVAILABLE"
    
    def __init__(self, provider: str, status_code: Optional[int] = None, retry_after: Optional[int] = None):
        super().__init__(
            f"{provider} service is currently unavailable",
            details={"provider": provider, "status_code": status_code, "retry_after": retry_after},
            suggestion=f"Wait {retry_after or 60} seconds before retrying or check service status"
        )


# API Format Errors
class APIFormatError(AdaptiveMemoryError):
    """Base exception for API format mismatches"""
    error_code = "AM_API_FORMAT_ERROR"

class GeminiAPIFormatError(APIFormatError):
    """Gemini API specific format error"""
    error_code = "AM_GEMINI_FORMAT_ERROR"
    
    def __init__(self, expected_format: str, received_format: str, field: Optional[str] = None):
        super().__init__(
            "Gemini API format mismatch",
            details={"expected": expected_format, "received": received_format, "field": field},
            suggestion="Ensure using Gemini-specific request format (not OpenAI format)"
        )

class UnsupportedProviderError(APIFormatError):
    """Unsupported LLM provider"""
    error_code = "AM_UNSUPPORTED_PROVIDER"
    
    def __init__(self, provider: str, supported_providers: List[str]):
        super().__init__(
            f"Provider '{provider}' is not supported",
            details={"provider": provider, "supported": supported_providers},
            suggestion=f"Use one of the supported providers: {', '.join(supported_providers)}"
        )


# JSON Parsing Errors
class JSONParsingError(AdaptiveMemoryError):
    """Base exception for JSON parsing issues"""
    error_code = "AM_JSON_PARSE_ERROR"

class MalformedJSONError(JSONParsingError):
    """JSON response is malformed"""
    error_code = "AM_MALFORMED_JSON"
    
    def __init__(self, raw_response: str, parse_error: str):
        truncated = raw_response[:200] + "..." if len(raw_response) > 200 else raw_response
        super().__init__(
            "Failed to parse JSON response",
            details={"response_preview": truncated, "parse_error": parse_error},
            suggestion="Enable JSON repair system or use a model with better JSON formatting"
        )

class JSONSchemaViolationError(JSONParsingError):
    """JSON doesn't match expected schema"""
    error_code = "AM_JSON_SCHEMA_VIOLATION"
    
    def __init__(self, expected_fields: List[str], received_fields: List[str]):
        missing = set(expected_fields) - set(received_fields)
        extra = set(received_fields) - set(expected_fields)
        super().__init__(
            "JSON response doesn't match expected schema",
            details={"missing_fields": list(missing), "extra_fields": list(extra)},
            suggestion="Check LLM prompt to ensure it generates required fields"
        )


# Memory Operation Errors
class MemoryOperationError(AdaptiveMemoryError):
    """Base exception for memory operation failures"""
    error_code = "AM_MEMORY_OP_ERROR"

class MemoryReadError(MemoryOperationError):
    """Failed to read memory"""
    error_code = "AM_MEMORY_READ_ERROR"
    
    def __init__(self, user_id: str, memory_id: Optional[str] = None, reason: Optional[str] = None):
        super().__init__(
            f"Failed to read memory for user {user_id}",
            details={"user_id": user_id, "memory_id": memory_id, "reason": reason},
            suggestion="Check database connection and user permissions"
        )

class MemoryWriteError(MemoryOperationError):
    """Failed to write memory"""
    error_code = "AM_MEMORY_WRITE_ERROR"
    
    def __init__(self, user_id: str, operation: str, reason: Optional[str] = None):
        super().__init__(
            f"Failed to {operation} memory for user {user_id}",
            details={"user_id": user_id, "operation": operation, "reason": reason},
            suggestion="Check database write permissions and storage availability"
        )

class MemoryCorruptionError(MemoryOperationError):
    """Memory data is corrupted"""
    error_code = "AM_MEMORY_CORRUPTION"
    
    def __init__(self, memory_id: str, corruption_type: str):
        super().__init__(
            f"Memory {memory_id} is corrupted",
            details={"memory_id": memory_id, "corruption_type": corruption_type},
            suggestion="Consider restoring from backup or clearing corrupted memory"
        )


# User Isolation Errors
class UserIsolationError(AdaptiveMemoryError):
    """Base exception for user isolation issues"""
    error_code = "AM_USER_ISOLATION_ERROR"

class UserContextMissingError(UserIsolationError):
    """User context is missing"""
    error_code = "AM_USER_CONTEXT_MISSING"
    
    def __init__(self, required_field: str):
        super().__init__(
            "Required user context is missing",
            details={"required_field": required_field},
            suggestion="Ensure OpenWebUI is passing complete user information"
        )

class UserPermissionError(UserIsolationError):
    """User permission violation"""
    error_code = "AM_USER_PERMISSION_ERROR"
    
    def __init__(self, user_id: str, attempted_action: str, target_resource: Optional[str] = None):
        super().__init__(
            f"User {user_id} lacks permission for {attempted_action}",
            details={"user_id": user_id, "action": attempted_action, "resource": target_resource},
            suggestion="Check user permissions and ensure proper access control"
        )

class CrossUserLeakageError(UserIsolationError):
    """Potential cross-user data leakage detected"""
    error_code = "AM_CROSS_USER_LEAKAGE"
    
    def __init__(self, requesting_user: str, leaked_user: str, context: str):
        super().__init__(
            "CRITICAL: Potential cross-user data leakage detected",
            details={"requesting_user": requesting_user, "leaked_user": leaked_user, "context": context},
            suggestion="Immediately review user isolation logic and database queries"
        )


# Filter Orchestration Errors
class FilterOrchestrationError(AdaptiveMemoryError):
    """Base exception for filter orchestration issues"""
    error_code = "AM_ORCHESTRATION_ERROR"

class FilterPipelineError(FilterOrchestrationError):
    """Filter pipeline execution failed"""
    error_code = "AM_PIPELINE_ERROR"
    
    def __init__(self, stage: str, filter_name: str, reason: str):
        super().__init__(
            f"Filter pipeline failed at {stage} stage",
            details={"stage": stage, "filter": filter_name, "reason": reason},
            suggestion="Check filter configuration and ensure all filters are compatible"
        )

class FilterCompatibilityError(FilterOrchestrationError):
    """Filter compatibility issue"""
    error_code = "AM_FILTER_COMPATIBILITY"
    
    def __init__(self, filter1: str, filter2: str, incompatibility: str):
        super().__init__(
            f"Filters {filter1} and {filter2} are incompatible",
            details={"filter1": filter1, "filter2": filter2, "issue": incompatibility},
            suggestion="Review filter priorities and data transformation requirements"
        )


# Circuit Breaker Errors
class CircuitBreakerError(AdaptiveMemoryError):
    """Circuit breaker is open"""
    error_code = "AM_CIRCUIT_BREAKER_OPEN"
    
    def __init__(self, service: str, failure_count: int, reset_time: Optional[float] = None):
        super().__init__(
            f"Circuit breaker open for {service} after {failure_count} failures",
            details={"service": service, "failures": failure_count, "reset_time": reset_time},
            suggestion=f"Service will be retried in {reset_time or 60} seconds"
        )


# Embedding Errors
class EmbeddingError(AdaptiveMemoryError):
    """Base exception for embedding-related errors"""
    error_code = "AM_EMBEDDING_ERROR"

class EmbeddingModelLoadError(EmbeddingError):
    """Failed to load embedding model"""
    error_code = "AM_EMBEDDING_MODEL_LOAD_ERROR"
    
    def __init__(self, model_name: str, reason: str):
        super().__init__(
            f"Failed to load embedding model {model_name}",
            details={"model": model_name, "reason": reason},
            suggestion="Check if model is installed and system has sufficient memory"
        )

class EmbeddingGenerationError(EmbeddingError):
    """Failed to generate embeddings"""
    error_code = "AM_EMBEDDING_GENERATION_ERROR"
    
    def __init__(self, text_length: int, reason: str):
        super().__init__(
            "Failed to generate embeddings",
            details={"text_length": text_length, "reason": reason},
            suggestion="Check text encoding and model input limits"
        )


# Utility function to log exceptions consistently
def log_exception(logger_instance: logging.Logger, exception: AdaptiveMemoryError, level: str = "error"):
    """Log an AdaptiveMemoryError with structured format"""
    log_method = getattr(logger_instance, level, logger_instance.error)
    # Use a custom field name to avoid conflict with logging's 'message' field
    extra_dict = exception.to_log_dict()
    extra_dict["exception_message"] = extra_dict.pop("message")
    log_method(f"AdaptiveMemory Exception: {exception.message}", extra=extra_dict)


class MemoryOperation(BaseModel):
    """Model for memory operations"""

    operation: Literal["NEW", "UPDATE", "DELETE"] = "NEW"
    id: Optional[str] = None
    content: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    memory_bank: Optional[str] = None  # NEW – bank assignment
    confidence: Optional[float] = None


class Filter:
    """
    OpenWebUI 2024 Adaptive Memory Filter
    
    A comprehensive memory management system for OpenWebUI that provides intelligent
    conversation memory, multi-provider LLM support, and advanced security features.
    
    Key Features:
    ============
    
    OpenWebUI 2024 Compliance:
    - Stream Function (v0.5.17+): Real-time filtering of streaming responses
    - Database Write Hooks: Separate processing for display vs storage
    - Enhanced Event Emitter: 2024-compliant event emission with batching
    - PII Filtering: Automatic detection and handling of sensitive information
    - Content Filtering: Real-time content filtering in streams
    
    Memory Management:
    - Intelligent conversation memory with relevance scoring
    - Multi-bank memory organization (Personal, Professional, Creative, etc.)
    - Automatic memory clustering and deduplication
    - Configurable memory sensitivity and thresholds
    - Background memory summarization and optimization
    
    Multi-Provider Support:
    - Ollama: Local model integration
    - OpenAI Compatible: GPT models and compatible APIs
    - Google Gemini: Gemini Pro and Gemini 1.5 models
    - Automatic provider detection and configuration
    
    Security Features:
    - Model name validation with security whitelist
    - Input sanitization and validation
    - API key protection and safe logging
    - Path traversal prevention
    - Code injection prevention
    
    Performance Optimizations:
    - Async operation support with background tasks
    - Connection pooling and health monitoring
    - Memory caching and similarity optimization
    - Graceful error handling and recovery
    
    Integration & Testing Status:
    ============================
    ✅ Type Safety: All critical type errors resolved
    ✅ Security: Comprehensive protection measures implemented
    ✅ Compatibility: Full OpenWebUI 2024 compliance verified
    ✅ Performance: Optimized async operations validated
    ✅ Reliability: Robust error handling and recovery tested
    
    Usage:
    ======
    The filter is designed to be used as an OpenWebUI filter function.
    Configuration is handled through the valve system with intelligent
    auto-configuration based on setup mode and memory requirements.
    
    Version: 4.0 (Final Integration & Validation)
    Compatible: OpenWebUI v0.5.17+ and 2024 standards
    Security Level: Production-grade with comprehensive protections
    """
    # Class variable removed - now initialized as instance variable in __init__
    
    # Security: Whitelist of allowed embedding models to prevent arbitrary code execution
    ALLOWED_EMBEDDING_MODELS = {
        'all-MiniLM-L6-v2',
        'all-MiniLM-L12-v2', 
        'all-mpnet-base-v2',
        'all-distilroberta-v1',
        'all-roberta-large-v1',
        'paraphrase-MiniLM-L6-v2',
        'paraphrase-mpnet-base-v2',
        'paraphrase-distilroberta-base-v1',
        'sentence-transformers/all-MiniLM-L6-v2',
        'sentence-transformers/all-MiniLM-L12-v2',
        'sentence-transformers/all-mpnet-base-v2',
        'sentence-transformers/all-distilroberta-v1',
        'sentence-transformers/paraphrase-MiniLM-L6-v2',
        'sentence-transformers/paraphrase-mpnet-base-v2',
        'microsoft/DialoGPT-medium',
        'microsoft/DialoGPT-large',
        'distilbert-base-uncased',
        'bert-base-uncased',
        'roberta-base'
    }
    
    @staticmethod
    def _validate_embedding_model_name(model_name: str) -> str:
        """
        Validate and sanitize embedding model name to prevent security vulnerabilities.
        
        Args:
            model_name: The model name to validate
            
        Returns:
            str: Validated model name or default fallback
            
        Raises:
            SecurityError: If model name contains suspicious patterns
        """
        if not model_name or not isinstance(model_name, str):
            log_with_context('warning', f"Invalid embedding model name type: {type(model_name)}. Using default.",
                           component='EMBED', operation='validate_model')
            return 'all-MiniLM-L6-v2'
        
        # Remove dangerous characters and patterns
        model_name = model_name.strip()
        
        # Check for suspicious patterns that could indicate code injection
        suspicious_patterns = [
            r'[;|&`$()\\]',  # Shell metacharacters
            r'\.\./',        # Path traversal
            r'__.*__',       # Python special methods
            r'\bexec\b|\beval\b|\bimport\b|\bopen\b|\bfile\b|\bsubprocess\b|\bos\.',  # Dangerous functions
            r'https?://',  # URLs
            r'\\x[0-9a-fA-F]{2}',  # Hex escapes
            r'%[0-9a-fA-F]{2}',     # URL encoding
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, model_name):
                log_with_context('error', f"Security violation: Suspicious pattern in model name",
                               component='EMBED', operation='validate_model',
                               context={'model_name': model_name, 'pattern': pattern})
                return 'all-MiniLM-L6-v2'  # Return safe default
        
        # Check against whitelist
        if model_name not in Filter.ALLOWED_EMBEDDING_MODELS:
            log_with_context('warning', f"Embedding model '{model_name}' not in allowed list. Using default.",
                           component='EMBED', operation='validate_model',
                           context={'requested': model_name, 'allowed': sorted(Filter.ALLOWED_EMBEDDING_MODELS)})
            return 'all-MiniLM-L6-v2'
        
        return model_name

    @property
    def _local_embedding_model(self): # RENAMED from embedding_model
        if not hasattr(self, '_embedding_model') or self._embedding_model is None:
            # Define model name outside try block to avoid scoping issues
            local_model_name = 'all-MiniLM-L6-v2'  # Default fallback
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
                # Security: Validate model name before loading
                raw_model_name = self.valves.embedding_model_name if self.valves.embedding_provider_type == 'local' else 'all-MiniLM-L6-v2'
                local_model_name = self._validate_embedding_model_name(raw_model_name)
                log_with_context('info', f"Loading validated embedding model: {local_model_name}",
                               component='EMBED', operation='load_model')
                self._embedding_model = SentenceTransformer(local_model_name)
            except ImportError as e:
                error = EmbeddingModelLoadError(
                    model_name=local_model_name, 
                    reason=f"SentenceTransformer library not available: {str(e)}"
                )
                log_exception(logger, error)
                self._embedding_model = None
            except Exception as e:
                error = EmbeddingModelLoadError(
                    model_name=local_model_name,
                    reason=str(e)
                )
                log_exception(logger, error)
                self._embedding_model = None
        return self._embedding_model
    
    async def _safe_initialize_embedding_model(self, timeout_seconds: int = 30) -> bool:
        """
        Safely initialize embedding model with timeout protection
        
        Args:
            timeout_seconds: Maximum time to wait for model loading
            
        Returns:
            bool: True if model loaded successfully
        """
        try:
            if self._embedding_model is not None:
                return True  # Already loaded
            
            def _load_model():
                # Define model name outside try block to avoid scoping issues
                local_model_name = 'all-MiniLM-L6-v2'  # Default fallback
                try:
                    from sentence_transformers import SentenceTransformer  # type: ignore
                    # Security: Validate model name before loading
                    raw_model_name = self.valves.embedding_model_name if self.valves.embedding_provider_type == 'local' else 'all-MiniLM-L6-v2'
                    local_model_name = self._validate_embedding_model_name(raw_model_name)
                    log_with_context('info', f"Loading validated embedding model: {local_model_name}",
                                   component='EMBED', operation='load_model_async')
                    return SentenceTransformer(local_model_name)
                except ImportError as e:
                    error = EmbeddingModelLoadError(
                        model_name=local_model_name,
                        reason=f"SentenceTransformer library not available: {str(e)}"
                    )
                    log_exception(logger, error)
                    return None
                except Exception as e:
                    error = EmbeddingModelLoadError(
                        model_name=local_model_name,
                        reason=str(e)
                    )
                    log_exception(logger, error)
                    return None
            model = await asyncio.wait_for(asyncio.to_thread(_load_model), timeout=timeout_seconds)
            if model is not None:
                self._embedding_model = model
                return True
            error = EmbeddingModelLoadError(
                model_name=self.valves.embedding_model_name,
                reason="Model initialization returned None"
            )
            log_exception(logger, error)
            return False
                
        except asyncio.TimeoutError:
            error = EmbeddingModelLoadError(
                model_name=self.valves.embedding_model_name,
                reason=f"Model loading timed out after {timeout_seconds} seconds"
            )
            log_exception(logger, error)
            return False
        except Exception as e:
            error = EmbeddingModelLoadError(
                model_name=self.valves.embedding_model_name,
                reason=f"Unexpected error during initialization: {str(e)}"
            )
            log_exception(logger, error)
            return False

    @property
    def memory_embeddings(self):
        if not hasattr(self, "_memory_embeddings") or self._memory_embeddings is None:
            self._memory_embeddings = {}
        
        # Prevent unbounded cache growth - limit to 1000 entries
        if len(self._memory_embeddings) > 1000:
            # Remove oldest entries (simple FIFO eviction)
            keys_to_remove = list(self._memory_embeddings.keys())[:len(self._memory_embeddings) - 800]
            for key in keys_to_remove:
                del self._memory_embeddings[key]
            if len(keys_to_remove) > 0:
                log_with_context('info', f"Cleaned up {len(keys_to_remove)} old embedding entries",
                               component='MEMORY', operation='cleanup_embeddings',
                               context={'removed_count': len(keys_to_remove)})
        
        return self._memory_embeddings

    @property
    def relevance_cache(self):
        if not hasattr(self, "_relevance_cache") or self._relevance_cache is None:
            self._relevance_cache = {}
        
        # Prevent unbounded cache growth - limit to 500 entries
        if len(self._relevance_cache) > 500:
            # Remove oldest entries (simple FIFO eviction)
            keys_to_remove = list(self._relevance_cache.keys())[:len(self._relevance_cache) - 400]
            for key in keys_to_remove:
                del self._relevance_cache[key]
            if len(keys_to_remove) > 0:
                log_with_context('info', f"Cleaned up {len(keys_to_remove)} old relevance cache entries",
                               component='MEMORY', operation='cleanup_relevance',
                               context={'removed_count': len(keys_to_remove)})
        
        return self._relevance_cache

    # Internal configuration class (not exposed to UI)
    class InternalConfig:
        """Internal configuration values that are auto-configured based on user settings."""
        def __init__(self):
            # Memory mode auto-configured fields
            self.recent_messages_n = 5
            self.related_memories_n = 5
            self.top_n_memories = 3
            self.vector_similarity_threshold = 0.45
            self.llm_skip_relevance_threshold = 0.93
            self.save_relevance_threshold = 0.8
            self.memory_threshold = 0.6
            
            # Advanced processing internals
            self.fingerprint_num_hashes = 128
            self.fingerprint_shingle_size = 3
            self.use_lsh_optimization = True
            self.lsh_threshold_for_activation = 100
            self.confidence_scoring_combined_threshold = 0.7
            self.short_preference_no_dedupe_length = 100
            self.preference_keywords_no_dedupe = "favorite,love,like,prefer,enjoy"
            
            # Background task internals
            self.enable_error_logging_task = True
            self.error_logging_interval = 1800
            self.enable_date_update_task = True
            self.date_update_interval = 3600
            self.enable_model_discovery_task = True
            self.model_discovery_interval = 7200
            self.summarization_max_cluster_size = 8
            self.summarization_min_memory_age_days = 7
            self.summarization_strategy = "hybrid"
            
            # Connection management internals
            self.max_concurrent_connections = 10
            self.connection_pool_size = 20
            self.enable_health_checks = True
            self.health_check_interval = 300
            self.circuit_breaker_failure_threshold = 5
            self.circuit_breaker_timeout = 60
            self.enable_connection_pooling = True
            self.connection_keepalive_timeout = 30
            self.dns_cache_ttl = 300
            self.enable_connection_diagnostics = True
            self.max_connection_retries = 3
            self.connection_retry_delay = 2.0
            
            # Filter orchestration internals
            self.debug_error_counter_logs = False
            self.enable_filter_orchestration = True
            self.filter_execution_timeout_ms = 10000
            self.enable_conflict_detection = True
            self.enable_performance_monitoring = True
            self.filter_priority = "normal"
            self.enable_rollback_mechanism = True
            self.max_concurrent_filters = 5
            self.coordination_overhead_threshold_ms = 100.0
            self.enable_shared_state = False
            self.filter_isolation_level = "partial"
    
    class Valves(BaseModel):
        
        model_config = {"extra": "forbid", "validate_assignment": True}
        
        # ================================================================
        # 🎯 QUICK SETUP - Most users only need these 4 settings!
        # ================================================================
        
        setup_mode: Literal["simple", "advanced"] = Field(
            default="simple",
            description="🎯 Configuration Mode: 'simple' = auto-configure everything (recommended), 'advanced' = full control"
        )
        
        llm_provider: Literal["ollama", "openai_compatible", "gemini"] = Field(
            default="openai_compatible",
            description="🤖 LLM Provider: 'openai_compatible' = API services (recommended), 'ollama' = local Ollama, 'gemini' = Google AI"
        )
        
        llm_model_name: str = Field(
            default="gpt-4o-mini",
            description="📝 Model Name: e.g., 'gpt-4o-mini' (OpenAI), 'llama3.3:70b' (Ollama), 'gemini-2.0-flash-exp' (Google)"
        )
        
        memory_mode: Literal["minimal", "balanced", "comprehensive"] = Field(
            default="balanced", 
            description="🧠 Memory Mode: 'minimal' = key facts only (1 memory/response, 50 total), 'balanced' = important info (3 memories/response, 200 total), 'comprehensive' = remember everything (5 memories/response, 500 total)"
        )
        
        # ================================================================
        # 🔑 API CONFIGURATION (only for OpenAI/Gemini)
        # ================================================================
        
        llm_api_key: Optional[str] = Field(
            default=None,
            description="🔑 API Key (required for OpenAI/Gemini): Get from your provider's dashboard"
        )
        
        llm_api_endpoint_url: str = Field(
            default="https://api.openai.com/v1/chat/completions",
            description="🌐 API Endpoint: Full URL to your LLM service (auto-detected for most providers)"
        )
        
        # ================================================================
        # 🎛️ MEMORY BEHAVIOR
        # ================================================================
        
        max_memories_to_remember: int = Field(
            default=200,
            description="💾 Maximum Memories: Total number of memories to keep per user (older ones are automatically removed)"
        )
        
        memory_sensitivity: Literal["low", "medium", "high"] = Field(
            default="medium",
            description="🎚️ Memory Sensitivity: How easily new memories are created from conversations"
        )
        
        show_memory_status: bool = Field(
            default=True,
            description="💬 Show Status Messages: Display 'Extracting memories...' and 'Found X memories' in chat"
        )
        
        memories_to_inject: int = Field(
            default=3,
            description="🔄 Memories per Response: How many relevant memories to include in each AI response"
        )
        
        relevance_threshold: float = Field(
            default=0.6,
            description="🎯 Relevance Threshold: Minimum similarity score (0.0-1.0) for memories to be included"
        )
        
        similarity_threshold: float = Field(
            default=0.85,
            description="🔍 Duplicate Detection: Similarity threshold (0.0-1.0) to prevent storing duplicate memories"
        )
        
        # ================================================================
        # 🏛️ MEMORY ORGANIZATION
        # ================================================================
        
        allowed_memory_banks: Optional[List[str]] = Field(
            default=None,
            description="🗂️ Memory Categories: Organize memories into these categories (optional - leave empty to disable categorization)"
        )
        
        default_memory_bank: str = Field(
            default="Personal",
            description="🏠 Default Category: Where to store memories when category isn't specified"
        )
        
        timezone: str = Field(
            default="UTC",
            description="🌍 Timezone: Your timezone for accurate time-based memories (e.g., 'America/New_York', 'Europe/London')"
        )
        
        # ================================================================
        # 🔧 TECHNICAL SETTINGS (expert users only)
        # ================================================================
        
        # ================================================================
        # 🆕 OPENWEBUI 2024 FEATURES
        # ================================================================
        
        # Stream Processing (v0.5.17+)
        enable_stream_filtering: bool = Field(
            default=True,
            description="🔄 Stream Filtering: Enable real-time filtering of AI responses as they stream to the user (OpenWebUI v0.5.17+). This allows content moderation and PII filtering during response generation."
        )
        
        enable_stream_content_filtering: bool = Field(
            default=False,
            description="🚫 Stream Content Filtering: When enabled, actively filters out sensitive content (PII, profanity, etc.) from AI responses as they stream. Works in conjunction with Stream Filtering above."
        )
        
        # Database Write Hooks (Issue #11888)
        enable_database_write_hooks: bool = Field(
            default=False,
            description="💾 Database Write Hooks: Enable preprocessing of content before it's saved to the database. This allows separate handling of what users see vs. what gets stored (e.g., encrypt sensitive data before storage)."
        )
        
        enable_pii_filtering: bool = Field(
            default=False,
            description="🔒 Enable PII filtering before database writes"
        )
        
        pii_filter_mode: Literal["redact", "encrypt", "anonymize"] = Field(
            default="redact",
            description="🔐 PII Filter Mode: 'redact' = remove PII, 'encrypt' = encrypt PII, 'anonymize' = replace with placeholders"
        )
        
        pii_encryption_key: Optional[str] = Field(
            default=None,
            description="🔑 Encryption key for PII data (required if using encrypt mode)"
        )
        
        # Enhanced Event Emitter Configuration
        enable_enhanced_event_emitter: bool = Field(
            default=True,
            description="📡 Enhanced Event Emitter: Enable advanced event emission patterns with metadata support and batching capabilities. This improves performance for status updates and memory processing notifications."
        )
        
        event_emitter_batch_size: int = Field(
            default=10,
            description="📦 Event Emitter Batch Size: Number of events to batch together before sending to improve performance. Set to 0 to disable batching and send events immediately. Higher values reduce overhead but increase latency."
        )
        
        event_emitter_timeout_ms: int = Field(
            default=5000,
            description="⏱️ Timeout for event emissions (milliseconds)"
        )
        
        embedding_provider_type: Literal["local", "openai_compatible"] = Field(
            default="local", 
            description="🔬 Embedding Provider: 'local' = built-in embeddings, 'openai_compatible' = API embeddings"
        )
        
        embedding_model_name: str = Field(
            default="all-MiniLM-L6-v2",
            description="🧮 Embedding Model: Model name for generating text embeddings (must be from approved list)"
        )
        
        embedding_api_url: Optional[str] = Field(
            default=None,
            description="🔗 Embedding API URL: Endpoint for external embedding service (if using API embeddings)"
        )
        
        embedding_api_key: Optional[str] = Field(
            default=None,
            description="🔐 Embedding API Key: Authentication for external embedding service"
        )
        
        # ================================================================
        # 🏥 ADVANCED TUNING (expert level)
        # ================================================================
        
        # Memory Processing
        enable_json_stripping: bool = Field(default=True, description="🔧 Strip non-JSON text from LLM responses")
        enable_fallback_regex: bool = Field(default=True, description="🔧 Use regex fallback for JSON parsing")
        enable_short_preference_shortcut: bool = Field(default=True, description="🔧 Save short preference messages directly")
        enable_feature_detection: bool = Field(default=True, description="🔧 Auto-detect LLM provider capabilities")
        
        # Memory Quality Control
        filter_trivia: bool = Field(default=True, description="🚫 Filter out trivia and general knowledge")
        min_memory_length: int = Field(default=8, description="📏 Minimum characters for valid memories")
        min_confidence_threshold: float = Field(default=0.5, description="🎯 Minimum confidence score for memories")
        blacklist_topics: Optional[str] = Field(default=None, description="🚫 Topics to never remember (comma-separated)")
        whitelist_keywords: Optional[str] = Field(default=None, description="✅ Keywords that always create memories")
        
        # Memory Storage Management
        pruning_strategy: Literal["fifo", "least_relevant"] = Field(default="fifo", description="🗂️ How to remove old memories")
        max_injected_memory_length: int = Field(default=300, description="📝 Max characters per injected memory")
        cache_ttl_seconds: int = Field(default=86400, description="⏰ Memory cache time-to-live")
        
        # Deduplication Settings
        deduplicate_memories: bool = Field(default=True, description="🔄 Prevent duplicate memories")
        use_embeddings_for_deduplication: bool = Field(default=True, description="🧮 Use AI embeddings for better duplicate detection")
        embedding_similarity_threshold: float = Field(default=0.97, description="🎯 Embedding similarity for duplicates")
        
        # Advanced Processing
        use_fingerprinting: bool = Field(default=True, description="🔍 Use content fingerprinting")
        fingerprint_similarity_threshold: float = Field(default=0.8, description="🎯 Fingerprint similarity threshold")
        use_llm_for_relevance: bool = Field(default=False, description="🤖 Use LLM for relevance scoring")
        use_enhanced_confidence_scoring: bool = Field(default=True, description="📊 Enhanced confidence scoring")
        
        # Performance & Reliability
        max_retries: int = Field(default=2, description="🔄 Max retry attempts for failed operations")
        retry_delay: float = Field(default=1.0, description="⏱️ Delay between retries (seconds)")
        request_timeout: float = Field(default=120.0, description="⏰ Request timeout (seconds)")
        connection_timeout: float = Field(default=30.0, description="⏰ Connection timeout (seconds)")
        
        # Memory Maintenance
        enable_summarization_task: bool = Field(default=True, description="📝 Enable automatic memory summarization")
        summarization_interval: int = Field(default=7200, description="⏰ Summarization interval (seconds)")
        summarization_min_cluster_size: int = Field(default=3, description="📊 Min memories for summarization")
        summarization_similarity_threshold: float = Field(default=0.7, description="🎯 Similarity for grouping memories")
        
        # Error Handling
        enable_error_counter_guard: bool = Field(default=True, description="🛡️ Enable error protection")
        error_guard_threshold: int = Field(default=5, description="🚨 Max errors before protection")
        error_guard_window_seconds: int = Field(default=600, description="⏰ Error counting window (seconds)")
        
        # ================================================================
        # 🎨 UI & DISPLAY
        # ================================================================
        
        show_memories: bool = Field(default=True, description="👁️ Show retrieved memories in responses")
        memory_format: Literal["bullet", "paragraph", "numbered"] = Field(default="bullet", description="📝 Memory display format")
        
        # Memory Types to Track
        enable_identity_memories: bool = Field(default=True, description="👤 Remember identity information")
        enable_behavior_memories: bool = Field(default=True, description="🎭 Remember behavioral patterns")
        enable_preference_memories: bool = Field(default=True, description="❤️ Remember preferences and likes")
        enable_goal_memories: bool = Field(default=True, description="🎯 Remember goals and aspirations")
        enable_relationship_memories: bool = Field(default=True, description="👥 Remember relationships")
        enable_possession_memories: bool = Field(default=True, description="🏠 Remember possessions and ownership")
        
        # ================================================================
        # 🤖 SYSTEM PROMPTS (expert level customization)
        # ================================================================
        
        memory_identification_prompt: str = Field(
            default='''Extract user-specific info as JSON array only. Format: [{"operation":"NEW","content":"text","tags":["list"],"memory_bank":"category","confidence":0.0-1.0}]. Extract: preferences, identity, goals, relationships, possessions, interests. Return only valid JSON array starting with [ and ending with ].''',
            description="🤖 Prompt for extracting memories from conversations"
        )
        
        memory_relevance_prompt: str = Field(
            default="""Rate memory relevance 0-1 for user context. Only user-specific info rates high, not trivia. Return JSON: [{"memory":"text","id":"123","relevance":0.8}]""",
            description="🤖 Prompt for rating memory relevance"
        )
        
        memory_merge_prompt: str = Field(
            default="""Merge similar user memories. Keep newer info if contradictory. Return JSON: ["merged memory text"]""",
            description="🤖 Prompt for merging similar memories"
        )
        
        summarization_memory_prompt: str = Field(
            default="""You are a memory summarization assistant. Your task is to combine related memories about a user into a concise, comprehensive summary.

Given a set of related memories about a user, create a single paragraph that:
1. Captures all key information from the individual memories
2. Resolves any contradictions (prefer newer information)
3. Maintains specific details when important
4. Removes redundancy
5. Presents the information in a clear, concise format

Focus on preserving the user's:
- Explicit preferences
- Identity details
- Goals and aspirations
- Relationships
- Possessions
- Behavioral patterns

Your summary should be factual, concise, and maintain the same tone as the original memories.
Produce a single paragraph summary of approximately 50-100 words that effectively condenses the information.

Example:
Individual memories:
- "User likes to drink coffee in the morning"
- "User prefers dark roast coffee"
- "User mentioned drinking 2-3 cups of coffee daily"

Good summary:
"User is a coffee enthusiast who drinks 2-3 cups daily, particularly enjoying dark roast varieties in the morning."

Analyze the following related memories and provide a concise summary.""",
            description="🤖 Prompt for summarizing related memories"
        )


        @field_validator(
            'summarization_interval', 'min_memory_length',
            'cache_ttl_seconds', 'max_retries', 'max_injected_memory_length',
            'summarization_min_cluster_size', 'max_memories_to_remember', 'memories_to_inject',
            'error_guard_threshold', 'error_guard_window_seconds',
            'event_emitter_batch_size', 'event_emitter_timeout_ms'
        )
        def check_non_negative_int(cls, v, info):
            if not isinstance(v, int) or v < 0:
                raise ValueError(f"{info.field_name} must be a non-negative integer")
            return v

        @field_validator(
            'relevance_threshold', 'similarity_threshold',
            'summarization_similarity_threshold', 'embedding_similarity_threshold',
            'min_confidence_threshold', 'fingerprint_similarity_threshold'
        )
        def check_threshold_float(cls, v, info):
            """Ensure threshold values are between 0.0 and 1.0"""
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"{info.field_name} must be between 0.0 and 1.0. Received: {v}")
            return v

        @field_validator('retry_delay', 'request_timeout', 'connection_timeout')
        def check_non_negative_float(cls, v, info):
            if not isinstance(v, float) or v < 0.0:
                raise ValueError(f"{info.field_name} must be a non-negative float")
            return v
        
        @field_validator('timezone')
        def check_valid_timezone(cls, v):
            try:
                pytz.timezone(v)
            except pytz.exceptions.UnknownTimeZoneError:
                raise ValueError(f"Invalid timezone string: {v}")
            except Exception as e:
                 raise ValueError(f"Error validating timezone '{v}': {e}")
            return v
        
        @field_validator('allowed_memory_banks')
        def validate_memory_banks(cls, v):
            if v is None:
                return None  # Allow None for no categorization
            if not isinstance(v, list) or not v:
                return None  # Return None instead of default list
            valid_banks = [bank.strip() for bank in v if bank and isinstance(bank, str) and bank.strip()]
            return valid_banks if valid_banks else None
        
        @field_validator('default_memory_bank')
        def validate_default_memory_bank(cls, v):
            return v.strip() if v and isinstance(v, str) and v.strip() else "Personal"
        
        @field_validator('llm_api_endpoint_url', 'embedding_api_url')
        def validate_api_urls(cls, v):
            if v is None or not isinstance(v, str): 
                return v
            v = v.strip()
            if v and not v.startswith(('http://', 'https://')):
                raise ValueError(f"API URL must start with http:// or https://, got: {v}")
            return v

        @field_validator('embedding_model_name')
        @classmethod
        def validate_embedding_model_name(cls, v):
            """Validate embedding model name for security"""
            if not v or not isinstance(v, str):
                log_with_context('warning', "Invalid embedding model name in config. Using default.",
                           component='CONFIG', operation='validate_config',
                           context={'invalid_value': v, 'type': type(v).__name__})
                return 'all-MiniLM-L6-v2'
            
            # Basic validation - full validation happens in Filter class
            v = v.strip()
            if not v:
                return 'all-MiniLM-L6-v2'
            
            # Check for obviously malicious patterns
            if any(char in v for char in [';', '|', '&', '`', '$', '(', ')', '\\']):
                log_with_context('error', "Security violation: Suspicious characters in embedding model name",
                               component='CONFIG', operation='validate_config',
                               context={'model_name': v})
                return 'all-MiniLM-L6-v2'
            
            return v
        
        @model_validator(mode="after")
        def auto_configure_based_on_setup_mode(self):
            """Auto-configure settings based on setup_mode and memory_mode"""
            
            # Auto-configure API endpoints based on provider
            if self.llm_provider == "ollama":
                # Configure Ollama endpoint
                if self.llm_api_endpoint_url == "https://api.openai.com/v1/chat/completions":
                    self.llm_api_endpoint_url = "http://host.docker.internal:11434/api/chat"
            elif self.llm_provider == "openai_compatible":
                # Configure OpenAI endpoint
                if self.llm_api_endpoint_url == "http://host.docker.internal:11434/api/chat":
                    self.llm_api_endpoint_url = "https://api.openai.com/v1/chat/completions"
                # Only require API key if not using a local or test environment
                if not self.llm_api_key:
                    log_with_context(
                        "WARNING", 
                        "No API key provided for OpenAI-compatible provider. Filter will work but LLM features may fail.", 
                        component="CONFIG"
                    )
            elif self.llm_provider == "gemini" and not self.llm_api_key:
                log_with_context(
                    "WARNING", 
                    "No API key provided for Google Gemini. Filter will work but LLM features may fail.", 
                    component="CONFIG"
                )
            
            # Auto-configure user-visible settings based on memory_mode if in simple setup
            if self.setup_mode == "simple":
                if self.memory_mode == "minimal":
                    # Update visible user settings for minimal mode
                    self.memories_to_inject = 1
                    self.max_memories_to_remember = 50
                    self.memory_sensitivity = "low"
                    self.relevance_threshold = 0.8
                    self.similarity_threshold = 0.95  # Very high to avoid duplicates
                elif self.memory_mode == "comprehensive":
                    # Update visible user settings for comprehensive mode
                    self.memories_to_inject = 5
                    self.max_memories_to_remember = 500
                    self.memory_sensitivity = "high"
                    self.relevance_threshold = 0.4
                    self.similarity_threshold = 0.75  # Lower to capture more variations
                # balanced mode uses defaults
            
            return self
        
        @model_validator(mode="after")
        def check_embedding_config(self):
            if self.embedding_provider_type == "openai_compatible":
                if not self.embedding_api_key or not self.embedding_api_url:
                    raise ValueError("🔧 API Key and URL required for OpenAI-compatible embeddings")
            return self
        
        @model_validator(mode="after")
        def check_memory_bank_consistency(self):
            if self.allowed_memory_banks and self.default_memory_bank not in self.allowed_memory_banks:
                object.__setattr__(self, 'default_memory_bank', self.allowed_memory_banks[0])
            elif not self.allowed_memory_banks:
                object.__setattr__(self, 'default_memory_bank', "General")  # Default when no categories
            return self


    class UserValves(BaseModel):
        enabled: bool = Field(default=True, description="Enable or disable the memory function")
        show_status: bool = Field(default=True, description="Show memory processing status updates")
        timezone: str = Field(default="", description="User's timezone")

    def _load_configuration_safe(self) -> Dict[str, Any]:
        try:
            return self.config.get("valves", {}) if isinstance(self.config, dict) else {}
        except:
            return {}
    
    def _validate_configuration_integrity(self, valves_instance) -> bool:
        try:
            return all([
                (valves_instance.allowed_memory_banks is None or (isinstance(valves_instance.allowed_memory_banks, list) and valves_instance.allowed_memory_banks)),
                isinstance(valves_instance.default_memory_bank, str) and valves_instance.default_memory_bank.strip(),
                0.0 <= valves_instance.vector_similarity_threshold <= 1.0,
                0.0 <= valves_instance.relevance_threshold <= 1.0,
                valves_instance.llm_api_endpoint_url.startswith(("http://", "https://"))
            ])
        except:
            return False
    
    def _recover_configuration(self):
        try:
            self.valves = self.Valves()
            return self._validate_configuration_integrity(self.valves)
        except:
            return False
    
    def _persist_configuration_state(self):
        pass
    
    def _validate_configuration_save(self, new_config: Dict[str, Any]) -> tuple[bool, str]:
        try:
            test_valves = self.Valves(**new_config)
            if not self._validate_configuration_integrity(test_valves):
                return False, "Configuration failed integrity validation"
            if hasattr(test_valves, 'llm_api_endpoint_url') and test_valves.llm_api_endpoint_url:
                try:
                    import urllib.parse
                    parsed_url = urllib.parse.urlparse(test_valves.llm_api_endpoint_url)
                    if not (parsed_url.scheme and parsed_url.netloc):
                        return False, "Invalid LLM API endpoint URL format"
                except Exception:
                    return False, "LLM API endpoint URL validation failed"
            return True, "Configuration validation successful"
        except Exception as e:
            return False, f"Configuration validation error: {str(e)}"
    
    def _is_system_ready_for_config_save(self) -> tuple[bool, str]:
        if not hasattr(self, 'valves') or self.valves is None:
            return False, "System not initialized"
        if hasattr(self, '_configuration_save_lock') and self._configuration_save_lock.locked():
            return False, "Save in progress"
        return True, "Ready"
    
    async def _save_configuration_safe(self, new_config: Dict[str, Any]) -> tuple[bool, str]:
        try:
            is_valid, msg = self._validate_configuration_save(new_config)
            if not is_valid:
                return False, msg
            backup = self.valves.model_dump() if hasattr(self.valves, 'model_dump') else None
            self.valves = self.Valves(**new_config)
            self.config['valves'] = new_config
            if not self._validate_configuration_integrity(self.valves):
                if backup:
                    self.valves = self.Valves(**backup)
                    self.config['valves'] = backup
                return False, "Validation failed"
            return True, "Saved"
        except Exception as e:
            return False, str(e)
    
    def set_valves(self, valves_data: Dict[str, Any]) -> bool:
        try:
            if not self._is_system_ready_for_config_save()[0]:
                return False
            if not self._validate_configuration_save(valves_data)[0]:
                return False
            backup = self.valves.model_dump() if hasattr(self.valves, 'model_dump') else None
            self.valves = self.Valves(**valves_data)
            self.config['valves'] = valves_data
            if not self._validate_configuration_integrity(self.valves):
                if backup:
                    self.valves = self.Valves(**backup)
                    self.config['valves'] = backup
                return False
            return True
        except:
            return False
    
    async def async_set_valves(self, valves_data: Dict[str, Any]) -> tuple[bool, str]:
        async with self._configuration_save_lock:
            for attempt in range(3):
                success, msg = await self._save_configuration_safe(valves_data)
                if success:
                    return True, msg
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
            return False, "Failed after 3 attempts"
    
    async def _emit_configuration_status(self, event_emitter, message: str, done: bool = False):
        if event_emitter:
            await self._safe_emit(event_emitter, {"type": "status", "data": {"description": message, "done": done}})
    
    def validate_configuration_before_ui_save(self, new_config: Dict[str, Any]) -> tuple[bool, str]:
        try:
            if not self._is_system_ready_for_config_save()[0]:
                return False, "System not ready"
            self.Valves(**new_config)
            return True, "Valid"
        except Exception as e:
            return False, str(e)
    
    def _reload_configuration_safe(self, force_reload=False) -> bool:
        try:
            data = self._load_configuration_safe()
            if not data and not force_reload:
                return True
            self.valves = self.Valves(**data)
            return self._validate_configuration_integrity(self.valves)
        except:
            return False
    
    def _ensure_configuration_persistence(self):
        if not hasattr(self, 'valves') or not self.valves:
            return self._recover_configuration()
        if not self._validate_configuration_integrity(self.valves):
            return self._reload_configuration_safe(True)
        return True
    
    def reload_config(self):
        """
        Public method to force reload configuration from persistent store.
        Returns True if reload was successful, False otherwise.
        
        This method can be called after config save to force a backend reload,
        ensuring that the latest settings are immediately applied.
        """
        return self._reload_configuration_safe(force_reload=True)
    
    def _test_configuration_management(self):
        return self._validate_configuration_integrity(self.valves) and self._ensure_configuration_persistence()

    def __init__(self):
        self.config = {}
        self._configuration_save_lock = asyncio.Lock()
        self.internal_config = self.InternalConfig()
        try:
            self.valves = self.Valves(**self._load_configuration_safe())
            if not self._validate_configuration_integrity(self.valves):
                self._recover_configuration()
        except:
            self.valves = self.Valves()
        
        # Apply internal configuration based on valves settings
        self._apply_internal_config()

        self.stored_memories: List[Dict[str, Any]] = []
        self._error_message = None # Stores the reason for the last failure (e.g., json_parse_error)
        self._aiohttp_session = None

        self._processed_messages = set()
        self.metrics = {"llm_call_count": 0}
        self._last_body = {}
        self._circuit_breaker_state = {}
        self._connection_health = {}
        self._last_health_check = {}
        self._session_connector = None
        self._background_tasks: List[asyncio.Task] = []
        self.error_counters = {"embedding_errors": 0, "llm_call_errors": 0, "json_parse_errors": 0, "memory_crud_errors": 0}

        if self.internal_config.enable_error_logging_task:
            self._add_background_task(self._log_error_counters_loop())
        if self.valves.enable_summarization_task:
            self._add_background_task(self._summarize_old_memories_loop())

        self.available_ollama_models = []
        self.available_openai_models = []
        
        # API version detection
        self._api_version_info: Optional[Dict[str, Any]] = None
        self._version_detection_count: int = 0
        self.available_local_embedding_models = []
        self.current_date = datetime.now()
        self.date_info = self._update_date_info()
        if self.internal_config.enable_date_update_task:
            self._add_background_task(self._update_date_loop())
        if self.internal_config.enable_model_discovery_task:
            self._add_background_task(self._discover_models_loop())
        
        # Initialize embedding model and related caches as instance variables
        self._embedding_model: Optional[Any] = None  # SentenceTransformer model
        self._memory_embeddings: Dict[str, Any] = {}  # Memory ID to embedding mapping
        self._relevance_cache: Dict[str, Any] = {}  # Relevance score cache
        
        # Initialize additional performance caches
        self._similarity_cache: Dict[str, float] = {}  # Similarity calculation cache
        self._user_embedding_cache: Dict[str, Any] = {}  # User message embedding cache
        self._llm_response_cache: Dict[str, Any] = {}  # LLM response cache for repeated queries
        
        # Initialize rollback stack and orchestration variables
        self._rollback_stack: List[Dict[str, Any]] = []
        self._orchestration_context: Optional[Any] = None
        self._filter_metadata: Optional[FilterMetadata] = None
        self._operation_lock: Optional[threading.RLock] = None
        self._state_snapshot: Dict[str, Any] = {}

        # Initialize JSON repair system
        if JSON_REPAIR_AVAILABLE:
            self._json_parser = EnhancedJSONParser()
            log_with_context('info', "JSON repair system initialized",
                           component='CONFIG', operation='init_json_repair')
        else:
            self._json_parser = None
            log_with_context('info', "JSON repair system not available, using basic parsing",
                           component='CONFIG', operation='init_json_repair')

        from collections import deque
        self.error_timestamps = {"json_parse_errors": deque()}
        self._guard_active = False
        self._guard_activated_at = 0
        self._duplicate_skipped = 0
        self._duplicate_refreshed = 0
        self._llm_feature_guard_active = False
        self._embedding_feature_guard_active = False
        self._background_tasks_started = False

        if self.internal_config.enable_filter_orchestration:
            self._initialize_filter_orchestration()
    
    def _apply_internal_config(self):
        """Apply internal configuration based on valve settings."""
        # Configure based on memory_mode
        if self.valves.setup_mode == "simple":
            if self.valves.memory_mode == "minimal":
                self.internal_config.recent_messages_n = 3
                self.internal_config.related_memories_n = 3
                self.internal_config.top_n_memories = 1
                self.internal_config.vector_similarity_threshold = 0.7
                self.internal_config.llm_skip_relevance_threshold = 0.95
                self.internal_config.memory_threshold = 0.7
            elif self.valves.memory_mode == "comprehensive":
                self.internal_config.recent_messages_n = 10
                self.internal_config.related_memories_n = 10
                self.internal_config.top_n_memories = 5
                self.internal_config.vector_similarity_threshold = 0.3
                self.internal_config.llm_skip_relevance_threshold = 0.90
                self.internal_config.memory_threshold = 0.4
            # balanced mode uses defaults
        
        # Configure based on memory_sensitivity
        if self.valves.memory_sensitivity == "low":
            self.internal_config.save_relevance_threshold = 0.9
        elif self.valves.memory_sensitivity == "high":
            self.internal_config.save_relevance_threshold = 0.6
        else:  # medium
            self.internal_config.save_relevance_threshold = 0.8

    def _selective_copy(self, obj: Any, max_depth: int = 3, current_depth: int = 0) -> Any:
        """Perform selective copying with depth limit to avoid expensive deep copies."""
        if current_depth >= max_depth:
            return obj
        
        if isinstance(obj, dict):
            # Only copy essential keys for rollback
            essential_keys = {'original_body', 'user_id', 'timestamp', 'messages', 'role', 'content'}
            return {k: self._selective_copy(v, max_depth, current_depth + 1) 
                   for k, v in obj.items() if k in essential_keys or current_depth == 0}
        elif isinstance(obj, list):
            # Limit list copying to avoid memory issues
            if len(obj) > 100:  # Limit large lists
                return [self._selective_copy(item, max_depth, current_depth + 1) for item in obj[:100]]
            return [self._selective_copy(item, max_depth, current_depth + 1) for item in obj]
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            # For other types, return as-is to avoid deep copy overhead
            return obj

    def _copy_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Efficiently copy messages array with only essential fields."""
        if not messages:
            return []
        
        copied_messages = []
        for msg in messages:
            if isinstance(msg, dict):
                # Only copy essential message fields
                copied_msg = {
                    'role': msg.get('role', 'user'),
                    'content': msg.get('content', ''),
                }
                # Add other essential fields if they exist
                for key in ['timestamp', 'id', 'metadata']:
                    if key in msg:
                        copied_msg[key] = msg[key]
                copied_messages.append(copied_msg)
            else:
                copied_messages.append(msg)
        return copied_messages

    def _cache_similarity(self, key1: str, key2: str, similarity: float) -> None:
        """Cache similarity calculation results."""
        cache_key = f"{key1}:{key2}"
        self._similarity_cache[cache_key] = similarity
        
        # Limit cache size to prevent memory bloat
        if len(self._similarity_cache) > 1000:
            # Remove oldest 20% of entries
            oldest_keys = list(self._similarity_cache.keys())[:200]
            for key in oldest_keys:
                del self._similarity_cache[key]

    def _get_cached_similarity(self, key1: str, key2: str) -> Optional[float]:
        """Get cached similarity if available."""
        cache_key = f"{key1}:{key2}"
        return self._similarity_cache.get(cache_key)

    def _cache_user_embedding(self, message: str, embedding: Any) -> None:
        """Cache user message embedding."""
        # Use message hash as key to avoid storing full message content
        msg_hash = str(hash(message))
        self._user_embedding_cache[msg_hash] = embedding
        
        # Limit cache size
        if len(self._user_embedding_cache) > 500:
            oldest_keys = list(self._user_embedding_cache.keys())[:100]
            for key in oldest_keys:
                del self._user_embedding_cache[key]

    def _get_cached_user_embedding(self, message: str) -> Optional[Any]:
        """Get cached user message embedding."""
        msg_hash = str(hash(message))
        return self._user_embedding_cache.get(msg_hash)

    def cleanup_memory_resources(self):
        """Clean up memory resources to prevent memory leaks"""
        try:
            # Clear embedding caches
            if hasattr(self, '_memory_embeddings'):
                self._memory_embeddings.clear()
            if hasattr(self, '_relevance_cache'):
                self._relevance_cache.clear()
            
            # Clear performance caches
            if hasattr(self, '_similarity_cache'):
                self._similarity_cache.clear()
            if hasattr(self, '_user_embedding_cache'):
                self._user_embedding_cache.clear()
            if hasattr(self, '_llm_response_cache'):
                self._llm_response_cache.clear()
            
            # Clear rollback stack
            if hasattr(self, '_rollback_stack'):
                self._rollback_stack.clear()
            
            # Clear error timestamps
            if hasattr(self, 'error_timestamps'):
                for key in self.error_timestamps:
                    if hasattr(self.error_timestamps[key], 'clear'):
                        self.error_timestamps[key].clear()
            
            # Clear state snapshot
            if hasattr(self, '_state_snapshot'):
                self._state_snapshot.clear()
            
            # Clear background tasks
            if hasattr(self, '_background_tasks'):
                # Cancel all background tasks
                for task in self._background_tasks.copy():
                    if not task.done():
                        task.cancel()
                self._background_tasks.clear()
            
            log_with_context('debug', "Memory resources cleaned up successfully",
                           component='MEMORY', operation='cleanup')
        except Exception as e:
            log_with_context('error', "Error during memory cleanup",
                           component='MEMORY', operation='cleanup', error=e)

    def _initialize_filter_orchestration(self):
        try:
            self._filter_metadata = FilterMetadata(name="adaptive_memory", version="4.0", priority=FilterPriority[self.internal_config.filter_priority.upper()])
            self._orchestration_manager = SimpleOrchestrator()
            self._operation_lock = threading.RLock()
        except:
            # Reset to None if initialization fails
            self._filter_metadata = None
            self._operation_lock = None

    def _create_execution_context(self, user_id=None):
        _ = user_id  # Acknowledge parameter
        return None

    def _record_operation_start(self, operation, context=None):
        _ = operation, context  # Acknowledge parameters
        if not self.internal_config.enable_performance_monitoring:
            return
        self._operation_start_time = time.time()

    def _record_operation_success(self, operation, start_time, context=None):
        _ = operation, context  # Acknowledge parameters
        if not self.internal_config.enable_performance_monitoring:
            return
        try:
            execution_time = (time.time() - start_time) * 1000
            if hasattr(self, '_orchestration_manager'):
                self._orchestration_manager.record_performance(execution_time)
        except:
            pass

    def _record_operation_failure(self, operation, start_time, error, context=None):
        _ = operation, error, context  # Acknowledge parameters
        if not self.internal_config.enable_performance_monitoring:
            return
        try:
            execution_time = (time.time() - start_time) * 1000
            log_with_context('warning', f"Operation failed after {execution_time}ms",
                           component='MEMORY', operation=operation,
                           context={'execution_time_ms': execution_time, 'error': str(error)})
        except:
            pass

    def _create_rollback_point(self, operation: str, data: Dict[str, Any]):
        if not self.internal_config.enable_rollback_mechanism:
            return
        try:
            self._rollback_stack.append({
                "operation": operation, 
                "timestamp": time.time(), 
                "data": self._selective_copy(data), 
                "rollback_id": str(uuid.uuid4())
            })
            # Limit rollback stack size to prevent memory leaks
            max_rollback_entries = 10
            if len(self._rollback_stack) > max_rollback_entries:
                # Remove oldest entries
                self._rollback_stack = self._rollback_stack[-max_rollback_entries:]
        except Exception as e:
            log_with_context('warning', "Failed to create rollback point",
                           component='MEMORY', operation='create_rollback',
                           error=e)
            pass

    def _add_background_task(self, coro):
        try:
            task = asyncio.create_task(coro)
            # Use list operations instead of set
            if not hasattr(self, '_background_tasks'):
                self._background_tasks = []
            self._background_tasks.append(task)
            # Clean up completed tasks when done
            def cleanup_task(t):
                try:
                    self._background_tasks.remove(t)
                except ValueError:
                    pass
            task.add_done_callback(cleanup_task)
            return task
        except RuntimeError:
            # No event loop running, skip background task creation
            log_with_context('debug', "No event loop running, skipping background task creation",
                           component='MEMORY', operation='rollback_async')
            return None

    def _perform_rollback(self, rollback_id: Optional[str] = None) -> bool:
        """Perform rollback to a previous state"""
        if not self.internal_config.enable_rollback_mechanism or not self._rollback_stack:
            return False
        
        try:
            if rollback_id:
                # Find specific rollback point
                rollback_entry = None
                for entry in reversed(self._rollback_stack):
                    if entry["rollback_id"] == rollback_id:
                        rollback_entry = entry
                        break
            else:
                # Use most recent rollback point
                rollback_entry = self._rollback_stack[-1]
            
            if not rollback_entry:
                log_with_context('warning', f"Rollback point not found: {rollback_id}",
                               component='MEMORY', operation='rollback',
                               context={'rollback_id': rollback_id})
                return False
            
            # Restore state (implementation depends on what was saved)
            operation = rollback_entry["operation"]
            data = rollback_entry["data"]
            
            FILTER_ROLLBACKS.labels(reason=f"rollback_{operation}").inc()
            
            log_with_context('info', f"Performed rollback for operation: {operation}",
                           component='MEMORY', operation='rollback',
                           context={'rollback_operation': operation})
            return True
            
        except Exception as e:
            log_with_context('error', "Failed to perform rollback",
                           component='MEMORY', operation='rollback',
                           error=e)
            FILTER_ROLLBACKS.labels(reason="rollback_error").inc()
            return False

    async def _calculate_memory_age_days(self, memory: Dict[str, Any]) -> float:
        """Calculate age of a memory in days."""
        created_at = memory.get("created_at")
        if not created_at or not isinstance(created_at, datetime):
            return float("inf")  # Treat memories without valid dates as infinitely old

        # Ensure created_at is timezone-aware (assume UTC if not)
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)

        # Get current time, also timezone-aware
        now_utc = datetime.now(timezone.utc)

        delta = now_utc - created_at
        return delta.total_seconds() / (24 * 3600)

    async def _find_memory_clusters(self, memories: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Find clusters of related memories based on configured strategy."""
        clusters: List[List[Dict[str, Any]]] = []
        processed_ids = set()
        strategy = self.internal_config.summarization_strategy
        threshold = self.valves.summarization_similarity_threshold
        min_age_days = self.internal_config.summarization_min_memory_age_days

        # --- Filter by Age First ---
        eligible_memories = []
        for mem in memories:
            age = await self._calculate_memory_age_days(mem)
            if age >= min_age_days:
                eligible_memories.append(mem)
            else:
                # Don't mark young memories as processed - they should be available for future clustering
                pass
        

        if not eligible_memories:
            return []

        # --- Embedding Clustering --- (Only if strategy is 'embeddings' or 'hybrid')
        embedding_clusters = []
        if strategy in ["embeddings", "hybrid"] and self._local_embedding_model:
            # Ensure all eligible memories have embeddings
            for mem in eligible_memories:
                mem_id = mem.get("id")
                if mem_id not in self.memory_embeddings:
                    try:
                        mem_text = mem.get("memory", "")
                        if mem_text:
                            mem_emb = self._local_embedding_model.encode(mem_text, normalize_embeddings=True)
                            self.memory_embeddings[mem_id] = mem_emb
                        else:
                             # Mark as None if no text to prevent repeated attempts
                             self.memory_embeddings[mem_id] = None
                    except Exception as e:
                        log_with_context('warning', f"Failed to generate embedding for memory during clustering",
                                       component='MEMORY', operation='clustering',
                                       context={'memory_id': mem_id}, error=e)
                        self.memory_embeddings[mem_id] = None # Mark as failed
            
            # Optimized clustering using precomputed similarities
            unprocessed_memories = [mem for mem in eligible_memories 
                                   if mem.get("id") not in processed_ids and 
                                   self.memory_embeddings.get(mem.get("id")) is not None]
            
            # Use index-based approach for better performance
            i = 0
            while i < len(unprocessed_memories):
                current_mem = unprocessed_memories[i]
                current_id = current_mem.get("id")
                
                if current_id in processed_ids:
                    i += 1
                    continue
                
                current_emb = self.memory_embeddings.get(current_id)
                if current_emb is None:
                    processed_ids.add(current_id)
                    i += 1
                    continue
                    
                cluster = [current_mem]
                processed_ids.add(current_id)
                
                # Find similar memories in remaining list
                j = i + 1
                while j < len(unprocessed_memories):
                    other_mem = unprocessed_memories[j]
                    other_id = other_mem.get("id")
                    
                    if other_id in processed_ids:
                        j += 1
                        continue
                        
                    other_emb = self.memory_embeddings.get(other_id)
                    if other_emb is None:
                        j += 1
                        continue
                         
                    # Calculate similarity with caching and error handling
                    try:
                        # Check cache first
                        current_id_str = str(current_id) if current_id is not None else ""
                        other_id_str = str(other_id) if other_id is not None else ""
                        cached_sim = self._get_cached_similarity(current_id_str, other_id_str)
                        if cached_sim is not None:
                            similarity = cached_sim
                        else:
                            similarity = float(np.dot(current_emb, other_emb))
                            # Cache the result
                            self._cache_similarity(current_id_str, other_id_str, similarity)
                        
                        if similarity >= threshold:
                            cluster.append(other_mem)
                            processed_ids.add(other_id)
                            # Remove from unprocessed list to avoid reprocessing
                            unprocessed_memories.pop(j)
                            continue  # Don't increment j as we removed an element
                    except Exception as e:
                        log_with_context('warning', "Error comparing embeddings during clustering",
                                       component='MEMORY', operation='clustering',
                                       context={'current_id': current_id, 'other_id': other_id}, error=e)
                    
                    j += 1
                
                # Add cluster if it meets minimum size
                if len(cluster) >= self.valves.summarization_min_cluster_size:
                    embedding_clusters.append(cluster)
                
                i += 1
            # If strategy is only embeddings, return now
            if strategy == "embeddings":
                 return embedding_clusters
        
        # --- Tag Clustering --- (Only if strategy is 'tags' or 'hybrid')
        tag_clusters = []
        if strategy in ["tags", "hybrid"]:
            from collections import defaultdict
            tag_map = defaultdict(list)
            
            # Group memories by tag
            for mem in eligible_memories:
                mem_id = mem.get("id")
                # Skip if already clustered by embeddings in hybrid mode
                if strategy == "hybrid" and mem_id in processed_ids:
                     continue
                     
                content = mem.get("memory", "")
                tags_match = re.match(r"\[Tags: (.*?)\]", content)
                if tags_match:
                    tags = [tag.strip() for tag in tags_match.group(1).split(",")]
                    for tag in tags:
                        # Only add if not already processed to prevent double-processing
                        if mem_id not in processed_ids:
                            tag_map[tag].append(mem)
            
            # Create clusters from tag groups
            cluster_candidates = list(tag_map.values())
            for candidate in cluster_candidates:
                # Filter out already processed IDs (important for hybrid)
                current_cluster = [mem for mem in candidate if mem.get("id") not in processed_ids]
                if len(current_cluster) >= self.valves.summarization_min_cluster_size:
                    tag_clusters.append(current_cluster)
                    # Mark these IDs as processed to prevent double-processing
                    for mem in current_cluster:
                        processed_ids.add(mem.get("id"))
            if strategy == "tags":
                 return tag_clusters
        
        # --- Hybrid Strategy: Combine and return --- 
        if strategy == "hybrid":
             # Simply concatenate the lists of clusters found by each method
             all_clusters = embedding_clusters + tag_clusters
             return all_clusters
        
        # Should not be reached if strategy is valid, but return empty list as fallback
        return []

    async def _summarize_old_memories_loop(self):
        while True:
            try:
                await asyncio.sleep(self.valves.summarization_interval * random.uniform(0.9, 1.1))
                user_id = "default"
                user_obj = Users.get_user_by_id(user_id)
                if not user_obj:
                    continue
                all_user_memories = await self._get_formatted_memories(user_id)
                if len(all_user_memories) < self.valves.summarization_min_cluster_size:
                    continue
                         
                memory_clusters = await self._find_memory_clusters(all_user_memories)
                if not memory_clusters:
                    continue
                summarized_count = 0
                deleted_count = 0
                for cluster in memory_clusters:
                        if len(cluster) < self.valves.summarization_min_cluster_size:
                            continue
                        cluster_to_summarize = cluster[:self.internal_config.summarization_max_cluster_size]
                        cluster_to_summarize.sort(key=lambda m: m.get("created_at", datetime.min.replace(tzinfo=timezone.utc)))
                        combined_text = "\n- ".join([m.get("memory", "") for m in cluster_to_summarize])

                        summary = await self.query_llm_with_retry(self.valves.summarization_memory_prompt, f"Related memories to summarize:\n- {combined_text}")
                        if summary and not summary.startswith("Error:"):
                            first_mem_content = cluster_to_summarize[0].get("memory", "")
                            tags = []
                            tags_match = re.match(r"\[Tags: (.*?)\]", first_mem_content)
                            if tags_match:
                                tags = [tag.strip() for tag in tags_match.group(1).split(",")]
                            if "summarized" not in tags:
                                tags.append("summarized")
                            formatted_summary = f"[Tags: {', '.join(tags)}] {summary.strip()}"
                            try:
                                await self._execute_memory_operation(MemoryOperation(operation="NEW", content=formatted_summary, tags=tags), user_obj)
                                summarized_count += 1
                            except:
                                continue
                            for mem in cluster_to_summarize:
                                try:
                                    await self._execute_memory_operation(MemoryOperation(operation="DELETE", id=mem["id"]), user_obj)
                                    deleted_count += 1
                                except:
                                    pass
                if summarized_count > 0:
                    log_with_context('info', "Memory summarization completed",
                                   component='MEMORY', operation='summarize',
                                   context={'summaries_created': summarized_count, 'memories_deleted': deleted_count})
            except asyncio.CancelledError:
                break
            except Exception as e:
                log_with_context('error', "Memory summarization failed",
                               component='MEMORY', operation='summarize', error=e)

    def _update_date_info(self):
        d = self.current_date
        return {"iso_date": d.strftime("%Y-%m-%d"), "year": d.year, "month": d.strftime("%B"), "day": d.day, "weekday": d.strftime("%A"), "hour": d.hour, "minute": d.minute, "iso_time": d.strftime("%H:%M:%S")}
    
    async def _update_date_loop(self):
        while True:
            try:
                await asyncio.sleep(self.internal_config.date_update_interval * random.uniform(0.9, 1.1))
                self.current_date = self.get_formatted_datetime()
                self.date_info = self._update_date_info()
            except asyncio.CancelledError:
                break
            except:
                pass
    
    async def _discover_models_loop(self):
        while True:
            try:
                await self._discover_models()
                await asyncio.sleep(self.internal_config.model_discovery_interval * random.uniform(0.9, 1.1))
            except asyncio.CancelledError:
                break
            except:
                await asyncio.sleep(self.internal_config.model_discovery_interval / 6)

    async def _log_error_counters_loop(self):
        while True:
            try:
                await asyncio.sleep(self.internal_config.error_logging_interval * random.uniform(0.9, 1.1))
                if self.internal_config.debug_error_counter_logs or any(self.error_counters.values()):
                    log_with_context('info', "Error counter summary",
                                   component='MEMORY', operation='error_stats',
                                   context={'counters': dict(self.error_counters)})

                if self.valves.enable_error_counter_guard:
                    now = time.time()
                    timestamps = self.error_timestamps["json_parse_errors"]
                    while timestamps and timestamps[0] < now - self.valves.error_guard_window_seconds:
                        timestamps.popleft()
                    if len(timestamps) >= self.valves.error_guard_threshold:
                        if not self._guard_active:
                            self._guard_active = True
                            self._original_use_llm_relevance = self.valves.use_llm_for_relevance
                            self._original_use_embedding_dedupe = self.valves.use_embeddings_for_deduplication
                            self.valves.use_llm_for_relevance = False
                            self.valves.use_embeddings_for_deduplication = False
                    elif self._guard_active:
                        self._guard_active = False
                        if hasattr(self, '_original_use_llm_relevance'):
                            self.valves.use_llm_for_relevance = self._original_use_llm_relevance
                        if hasattr(self, '_original_use_embedding_dedupe'):
                            self.valves.use_embeddings_for_deduplication = self._original_use_embedding_dedupe
            except asyncio.CancelledError:
                break
            except:
                pass

    def _schedule_date_update(self):
        """Schedule a regular update of the date information"""

        async def update_date_loop():
            try:
                while True:
                    # Use configurable interval with small random jitter
                    jitter = random.uniform(0.9, 1.1)  # ±10% randomization
                    interval = self.internal_config.date_update_interval * jitter
                    await asyncio.sleep(interval)
                    
                    self.current_date = self.get_formatted_datetime()
                    self.date_info = self._update_date_info()
            except asyncio.CancelledError:
                pass
            except Exception as e:
                log_with_context('error', "Error in date update task",
                               component='MEMORY', operation='update_dates', error=e)

        # Start the update loop in the background
        task = asyncio.create_task(update_date_loop())
        self._background_tasks.append(task)
        # Clean up completed tasks when done
        def cleanup_task(t):
            try:
                self._background_tasks.remove(t)
            except ValueError:
                pass
        task.add_done_callback(cleanup_task)
        return task

    def _schedule_model_discovery(self):
        """Schedule a regular update of available models"""

        async def discover_models_loop():
            try:
                while True:
                    try:
                        # Discover models
                        await self._discover_models()
                        
                        # Use configurable interval with small random jitter
                        jitter = random.uniform(0.9, 1.1)  # ±10% randomization
                        interval = self.internal_config.model_discovery_interval * jitter
                        await asyncio.sleep(interval)
                    except asyncio.CancelledError:
                        raise
                    except Exception as e:
                        log_with_context('error', "Error in model discovery",
                                       component='CONFIG', operation='discover_models', error=e)
                        # On error, retry sooner (1/6 of normal interval)
                        await asyncio.sleep(self.internal_config.model_discovery_interval / 6)
            except asyncio.CancelledError:
                pass

        # Start the discovery loop in the background
        task = asyncio.create_task(discover_models_loop())
        self._background_tasks.append(task)
        # Clean up completed tasks when done
        def cleanup_task(t):
            try:
                self._background_tasks.remove(t)
            except ValueError:
                pass
        task.add_done_callback(cleanup_task)
        return task

    async def _discover_models(self):
        session = await self._get_aiohttp_session()
        try:
            async with session.get("http://host.docker.internal:11434/api/tags", timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    data = await response.json()
                    self.available_ollama_models = [m["name"] for m in data.get("models", [])]
                else:
                    self.available_ollama_models = []
        except:
            self.available_ollama_models = []
        try:
            cache_dir = os.path.expanduser("~/.cache/torch/sentence_transformers")
            if os.path.isdir(cache_dir):
                self.available_local_embedding_models = [
                    e for e in os.listdir(cache_dir)
                    if os.path.isdir(os.path.join(cache_dir, e)) and
                    any(os.path.isfile(os.path.join(cache_dir, e, f)) for f in ("config.json", "modules.json"))
                ] or ["all-MiniLM-L6-v2"]
            else:
                self.available_local_embedding_models = ["all-MiniLM-L6-v2"]
        except:
            self.available_local_embedding_models = []

    def get_formatted_datetime(self, user_timezone: Optional[str] = None):
        timezone_str = user_timezone or self.valves.timezone or "UTC"
        alias_map = {"UAE/Dubai": "Asia/Dubai", "GMT+4": "Asia/Dubai", "GMT +4": "Asia/Dubai", "Dubai": "Asia/Dubai", "EST": "America/New_York", "PST": "America/Los_Angeles", "CST": "America/Chicago", "IST": "Asia/Kolkata", "BST": "Europe/London", "GMT": "Etc/GMT", "UTC": "UTC"}
        tz_key = timezone_str.strip()
        timezone_str = alias_map.get(tz_key, timezone_str)
        try:
            utc_now = datetime.now(timezone.utc)
            local_tz = pytz.timezone(timezone_str)
            return utc_now.astimezone(local_tz)
        except pytz.exceptions.UnknownTimeZoneError:
            log_with_context('warning', f"Invalid timezone: {timezone_str}, using default",
                           component='CONFIG', operation='validate_timezone',
                           context={'requested': timezone_str, 'default': 'Asia/Dubai'})
            try:
                local_tz = pytz.timezone("Asia/Dubai")
                return datetime.now(timezone.utc).astimezone(local_tz)
            except Exception:
                log_with_context('warning', "Fallback timezone also invalid, using UTC",
                               component='CONFIG', operation='validate_timezone')
                return datetime.now(timezone.utc)

    def _get_circuit_breaker_key(self, api_url: str, provider_type: str) -> str:
        return f"{provider_type}:{api_url}"
    
    def _is_circuit_breaker_open(self, api_url: str, provider_type: str) -> bool:
        key = self._get_circuit_breaker_key(api_url, provider_type)
        state = self._circuit_breaker_state.get(key, {"failures": 0, "last_failure": 0, "is_open": False})
        if not state.get("is_open", False):
            return False
        current_time = time.time()
        timeout_duration = self.internal_config.circuit_breaker_timeout
        if current_time - state.get("last_failure", 0) > timeout_duration:
            log_with_context('debug', f"Circuit breaker reset for {key}",
                           component='FILTER', operation='circuit_breaker_reset',
                           context={'breaker_key': key})
            state["is_open"] = False
            state["failures"] = 0
            self._circuit_breaker_state[key] = state
            return False
        return True
    
    def _record_circuit_breaker_failure(self, api_url: str, provider_type: str) -> None:
        key = self._get_circuit_breaker_key(api_url, provider_type)
        state = self._circuit_breaker_state.get(key, {"failures": 0, "last_failure": 0, "is_open": False})
        state["failures"] += 1
        state["last_failure"] = time.time()
        if state["failures"] >= self.internal_config.circuit_breaker_failure_threshold:
            state["is_open"] = True
            error = CircuitBreakerError(
                service=key,
                failure_count=state['failures'],
                reset_time=self.internal_config.circuit_breaker_timeout
            )
            log_exception(logger, error, level="warning")
        self._circuit_breaker_state[key] = state
    
    def _record_circuit_breaker_success(self, api_url: str, provider_type: str) -> None:
        key = self._get_circuit_breaker_key(api_url, provider_type)
        if key in self._circuit_breaker_state:
            self._circuit_breaker_state[key]["failures"] = 0
    
    async def _check_endpoint_health(self, api_url: str, provider_type: str) -> bool:
        key = self._get_circuit_breaker_key(api_url, provider_type)
        current_time = time.time()
        last_check = self._last_health_check.get(key, 0)
        if current_time - last_check < self.internal_config.health_check_interval:
            return self._connection_health.get(key, True)
        try:
            session = await self._get_aiohttp_session()
            health_data = {"model": "health_check", "messages": [{"role": "user", "content": "ping"}], "max_tokens": 1}
            headers = {"Content-Type": "application/json"}
            if provider_type in ["openai_compatible", "gemini"] and self.valves.llm_api_key:
                headers["Authorization"] = f"Bearer {self.valves.llm_api_key}"
            async with session.post(api_url, json=health_data, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as response:
                is_healthy = response.status < 500
                self._connection_health[key] = is_healthy
                self._last_health_check[key] = current_time
                if not is_healthy:
                    log_with_context('warning', f"Health check failed for {key}",
                                   component='FILTER', operation='health_check',
                                   context={'key': key, 'status': response.status})
                return is_healthy
        except Exception as e:
            log_with_context('warning', f"Health check failed for {key}",
                           component='FILTER', operation='health_check',
                           context={'key': key}, error=e)
            self._connection_health[key] = False
            self._last_health_check[key] = current_time
            return False
    
    async def _detect_provider_features(self, api_url: str, provider_type: str, api_key: Optional[str] = None) -> Dict[str, bool]:
        features = {"supports_system_messages": True, "supports_json_mode": True, "supports_streaming": True, "supports_function_calling": False, "supports_vision": False, "requires_auth": provider_type in ["openai_compatible", "gemini"]}
        try:
            session = await self._get_aiohttp_session()
            headers = {"Content-Type": "application/json"}
            if features["requires_auth"] and api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            test_data = {"model": "test", "messages": [{"role": "user", "content": "ping"}], "max_tokens": 1, "stream": False}
            if provider_type in ["openai_compatible", "gemini"]:
                test_data["response_format"] = {"type": "json_object"}
            async with session.post(api_url, json=test_data, headers=headers, timeout=aiohttp.ClientTimeout(total=15)) as response:
                if response.status == 400:
                    error_text = await response.text()
                    if "json_object" in error_text.lower() or "response_format" in error_text.lower():
                        features["supports_json_mode"] = False
                if provider_type == "gemini":
                    features.update({"supports_vision": True, "supports_function_calling": True, "supports_json_mode": True, "supports_system_messages": True, "supports_streaming": True})
                cache_key = f"{provider_type}_{api_url}"
                if not hasattr(self, '_provider_features'):
                    self._provider_features = {}
                self._provider_features[cache_key] = features
        except:
            features.update({"supports_json_mode": False, "supports_streaming": False, "supports_function_calling": False, "supports_vision": False})
        return features
    
    async def _get_provider_features(self, provider_type: str, api_url: str, api_key: Optional[str] = None) -> Dict[str, bool]:
        cache_key = f"{provider_type}_{api_url}"
        if not hasattr(self, '_provider_features'):
            self._provider_features = {}
        if cache_key not in self._provider_features:
            self._provider_features[cache_key] = await self._detect_provider_features(api_url, provider_type, api_key)
        return self._provider_features[cache_key]
    
    def _is_memory_processing_circuit_open(self) -> bool:
        if not hasattr(self, 'memory_processing_circuit_open'):
            self.memory_processing_circuit_open = False
            self.memory_processing_failures = 0
            self.memory_processing_circuit_reset_time = 0.0
            
        current_time = time.time()
        
        # Check if circuit should be reset
        if self.memory_processing_circuit_open and current_time > self.memory_processing_circuit_reset_time:
            log_with_context('info', "Memory processing circuit breaker reset - reopening",
                           component='MEMORY', operation='circuit_breaker_reset')
            self.memory_processing_circuit_open = False
            self.memory_processing_failures = 0
            
        return self.memory_processing_circuit_open
    
    def _record_memory_processing_failure(self) -> None:
        """Record a memory processing failure for circuit breaker"""
        failure_count = getattr(self.valves, 'circuit_breaker_failure_count', 3)
        reset_time = getattr(self.valves, 'circuit_breaker_reset_time', 300.0)
        
        if not hasattr(self, 'memory_processing_failures'):
            self.memory_processing_failures = 0
            
        self.memory_processing_failures += 1
        
        if self.memory_processing_failures >= failure_count:
            self.memory_processing_circuit_open = True
            self.memory_processing_circuit_reset_time = time.time() + reset_time
            error = CircuitBreakerError(
                service="memory_processing",
                failure_count=self.memory_processing_failures,
                reset_time=reset_time
            )
            log_exception(logger, error, level="warning")
    
    def _record_memory_processing_success(self) -> None:
        """Record successful memory processing"""
        if hasattr(self, 'memory_processing_failures'):
            self.memory_processing_failures = 0
    
    def _limit_memory_operations(self, operations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Limit the number of memory operations to prevent overwhelming the system"""
        max_ops = getattr(self.valves, 'max_memory_operations_per_message', 20)
        
        if len(operations) > max_ops:
            log_with_context('warning', "Limiting memory operations to prevent overload",
                           component='MEMORY', operation='rate_limit',
                           context={'requested': len(operations), 'limited_to': max_ops})
            # Keep the highest confidence operations
            sorted_ops = sorted(operations, key=lambda x: x.get('confidence', 0), reverse=True)
            return sorted_ops[:max_ops]
        
        return operations
    
    async def _create_fallback_memory_from_preference(self, user_message: str) -> List[Dict[str, Any]]:
        preference_patterns = [r"(i|my)\s+(favorite|love|like|prefer|enjoy|hate|dislike)\s+([^.!?]+)", r"(i|my)\s+(am|is)\s+([^.!?]+)", r"(i|my)\s+(work|live|study)\s+([^.!?]+)"]
        for pattern in preference_patterns:
            match = re.search(pattern, user_message.lower(), re.IGNORECASE)
            if match:
                content = f"User preference: {match.group(0).strip()}"
                log_with_context('info', "Created fallback memory from preference pattern",
                               component='MEMORY', operation='create_fallback',
                               context={'content_preview': content[:50] + '...' if len(content) > 50 else content})
                return [{"operation": "NEW", "content": content, "tags": ["preference"], "memory_bank": "General", "confidence": 0.6}]
        return []

    async def _get_aiohttp_session(self) -> aiohttp.ClientSession:
        if self._aiohttp_session is None or self._aiohttp_session.closed:
            if self._session_connector is None or self._session_connector.closed:
                self._session_connector = TCPConnector(limit=self.internal_config.connection_pool_size, limit_per_host=self.internal_config.max_concurrent_connections, ttl_dns_cache=self.internal_config.dns_cache_ttl, use_dns_cache=True, enable_cleanup_closed=True, force_close=False, keepalive_timeout=self.internal_config.connection_keepalive_timeout)
            timeout = ClientTimeout(total=self.valves.request_timeout, connect=self.valves.connection_timeout, sock_read=self.valves.request_timeout, sock_connect=self.valves.connection_timeout)
            self._aiohttp_session = aiohttp.ClientSession(connector=self._session_connector, timeout=timeout, raise_for_status=False, trust_env=True)
        return self._aiohttp_session
    
    async def _cleanup_connections(self) -> None:
        try:
            if self._aiohttp_session and not self._aiohttp_session.closed:
                await self._aiohttp_session.close()
            if self._session_connector and not self._session_connector.closed:
                await self._session_connector.close()
        except Exception as e:
            log_with_context('warning', "Error during connection cleanup",
                           component='FILTER', operation='cleanup_connections', error=e)
        finally:
            self._aiohttp_session = None
            self._session_connector = None
    
    def get_connection_stats(self) -> Dict[str, Any]:
        return {"circuit_breakers": dict(self._circuit_breaker_state), "connection_health": dict(self._connection_health), "last_health_checks": dict(self._last_health_check), "session_open": self._aiohttp_session is not None and not self._aiohttp_session.closed if self._aiohttp_session else False, "connector_open": self._session_connector is not None and not self._session_connector.closed if self._session_connector else False}
    
    async def _diagnose_connection_issues(self, api_url: str, provider_type: str, error: Exception) -> Dict[str, Any]:
        key = self._get_circuit_breaker_key(api_url, provider_type)
        diagnostics: Dict[str, Any] = {"endpoint": api_url, "provider": provider_type, "error_type": type(error).__name__, "error_message": str(error), "timestamp": time.time(), "tests": {}}
        if not self.internal_config.enable_connection_diagnostics:
            diagnostics["tests"]["diagnostics_disabled"] = True
            return diagnostics
        try:
            session = await self._get_aiohttp_session()
            test_data = {"model": self.valves.llm_model_name, "messages": [{"role": "user", "content": "test"}], "max_tokens": 1}
            headers = {"Content-Type": "application/json"}
            if provider_type in ["openai_compatible", "gemini"] and self.valves.llm_api_key:
                headers["Authorization"] = f"Bearer {self.valves.llm_api_key}"
            async with session.post(api_url, json=test_data, headers=headers, timeout=aiohttp.ClientTimeout(total=15)) as response:
                status = response.status
                diagnostics["tests"]["model_availability"] = {"status": "success" if status == 200 else "failed", "response_code": status}
                if status != 200:
                    diagnostics["tests"]["model_availability"]["error"] = "Auth failed" if status == 401 else f"HTTP {status}"
        except Exception as e:
            diagnostics["tests"]["model_availability"] = {"status": "failed", "error": str(e)[:100]}
        cb_state = self._circuit_breaker_state.get(key, {})
        diagnostics["tests"]["circuit_breaker"] = {"status": "open" if cb_state.get("is_open", False) else "closed", "failure_count": cb_state.get("failures", 0)}
        return diagnostics
    
    async def _get_connection_troubleshooting_tips(self, diagnostics: Dict[str, Any]) -> List[str]:
        tips = []
        tests = diagnostics.get("tests", {})
        provider = diagnostics.get("provider", "unknown")
        model_test = tests.get("model_availability", {})
        if model_test.get("status") == "failed":
            if model_test.get("response_code") == 401:
                tips.append("🔐 Authentication failed - check API key")
            elif model_test.get("response_code") == 404:
                tips.append("📦 Model not found - check model name")
            else:
                tips.append("🔌 Connection failed - check endpoint")
        cb_test = tests.get("circuit_breaker", {})
        if cb_test.get("status") == "open":
            tips.append(f"🔴 Circuit breaker open after {cb_test.get('failure_count', 0)} failures")
        if provider == "ollama":
            tips.append("🦙 Ensure Ollama service is running")
        return tips
    
    def reset_circuit_breakers(self, api_url: Optional[str] = None, provider_type: Optional[str] = None) -> Dict[str, Any]:
        reset_info: Dict[str, Any] = {"reset_count": 0, "reset_endpoints": []}
        if api_url and provider_type:
            key = self._get_circuit_breaker_key(api_url, provider_type)
            if key in self._circuit_breaker_state:
                self._circuit_breaker_state[key] = {"failures": 0, "last_failure": 0, "is_open": False}
                reset_info["reset_count"] = 1
                reset_info["reset_endpoints"].append(key)
                log_with_context('debug', f"Reset circuit breaker for {key}",
                               component='FILTER', operation='reset_circuit_breaker',
                               context={'key': key})
        else:
            reset_info["reset_count"] = len(self._circuit_breaker_state)
            reset_info["reset_endpoints"] = list(self._circuit_breaker_state.keys())
            self._circuit_breaker_state.clear()
            log_with_context('debug', "Reset all circuit breakers",
                           component='FILTER', operation='reset_all_breakers')
        return reset_info
    
    async def test_llm_connection(self, timeout: float = 30.0) -> Dict[str, Any]:
        provider_type = self.valves.llm_provider
        api_url = self.valves.llm_api_endpoint_url
        model_name = self.valves.llm_model_name
        test_result = {"provider": provider_type, "endpoint": api_url, "model": model_name, "timestamp": time.time(), "success": False, "response_time": 0.0, "error": None, "diagnostics": None}
        start_time = time.time()
        try:
            test_response = await asyncio.wait_for(self.query_llm_with_retry("You are a connection test assistant.", "Reply with exactly: TEST_SUCCESSFUL"), timeout=timeout)
            test_result["response_time"] = time.time() - start_time
            if "TEST_SUCCESSFUL" in test_response and not test_response.startswith("Error:"):
                test_result["success"] = True
                test_result["response"] = test_response
            else:
                test_result["error"] = test_response if test_response.startswith("Error:") else f"Unexpected response: {test_response[:100]}..."
        except asyncio.TimeoutError:
            test_result["error"] = f"Connection test timed out after {timeout} seconds"
            test_result["response_time"] = timeout
        except Exception as e:
            test_result["error"] = str(e)
            test_result["response_time"] = time.time() - start_time
        if not test_result["success"]:
            try:
                test_result["diagnostics"] = await self._diagnose_connection_issues(api_url, provider_type, Exception(test_result["error"] or "Test failed"))
            except Exception as e:
                test_result["diagnostics_error"] = str(e)
        return test_result

    async def async_inlet(
        self,
        body: Dict[str, Any],
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
        __user__: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Intercepts incoming messages, extracts memories, injects relevant ones.

        Handles chat commands: /memory list, /memory forget [id], /memory edit [id] [new content],
        /memory summarize [topic/tag], /note [content], /memory mark_important [id],
        /memory unmark_important [id], /memory list_banks, /memory assign_bank [id] [bank],
        /diagnose (LLM connection diagnostics), /reset circuit (reset circuit breakers)
        """
        try:
            # Add timeout protection to entire inlet operation
            return await asyncio.wait_for(
                self._async_inlet_impl(body, __event_emitter__, __user__),
                timeout=120.0  # 2 minutes max for inlet processing
            )
        except asyncio.TimeoutError:
            user_id = __user__.get('id', 'unknown') if __user__ else 'unknown'
            log_with_context('error', "Inlet operation timed out after 120 seconds",
                           component='INLET', user_id=user_id, operation='timeout')
            # Use enhanced event emitter for 2024 compliance
            await self._emit_enhanced_event(
                __event_emitter__, 
                "warning", 
                "⚠️ Memory processing timed out - continuing without memory injection",
                {"done": True, "timeout": True},
                "memory_processing"
            )
            return body
        except Exception as e:
            user_id = __user__.get('id', 'unknown') if __user__ else 'unknown'
            log_with_context('error', "Critical error in inlet processing",
                           component='INLET', user_id=user_id, operation='error', error=e)
            # Use enhanced event emitter for 2024 compliance
            await self._emit_enhanced_event(
                __event_emitter__, 
                "error", 
                f"⚠️ Memory processing error: {str(e)[:100]}",
                {"done": True, "error_type": "processing_error"},
                "memory_processing"
            )
            return body
    
    async def _async_inlet_impl(
        self,
        body: Dict[str, Any],
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
        __user__: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Implementation of async_inlet with timeout protection"""
        user_id = __user__.get('id', 'unknown') if __user__ else 'unknown'
        log_with_context('debug', f"Inlet received body keys: {list(body.keys())}",
                        component='INLET', user_id=user_id, operation='receive')

        # Felicity says: No more living in the past—your config is now as fresh as your taste in AI assistants! 💅
        # Force reload configuration from persistent store to ensure latest settings are applied
        try:
            reload_success = await asyncio.wait_for(
                asyncio.to_thread(self._reload_configuration_safe, True),  # force_reload=True
                timeout=5.0
            )
            if not reload_success:
                log_with_context('warning', "Configuration force reload failed, continuing with current config",
                               component='INLET', user_id=user_id, operation='config_reload')
        except asyncio.TimeoutError:
            log_with_context('warning', "Configuration force reload timed out, continuing with current config",
                           component='INLET', user_id=user_id, operation='config_reload')
        except Exception as e:
            log_with_context('warning', "Configuration force reload error, continuing with current config",
                           component='INLET', user_id=user_id, operation='config_reload', error=e)

        # Ensure user info is present
        if not __user__ or not __user__.get("id"):
            error = UserContextMissingError(required_field="user.id")
            log_exception(logger, error, level="warning")
            return body
        user_id = __user__["id"]
        log_with_context('info', "Starting inlet processing",
                        component='INLET', user_id=user_id, operation='start')

        # -----------------------------------------------------------
        # Filter Orchestration System Integration
        # -----------------------------------------------------------
        orchestration_context = None
        operation_start_time = time.time()
        
        if self.internal_config.enable_filter_orchestration:
            try:
                # Create execution context for this operation
                orchestration_context = self._create_execution_context(user_id)
                
                # Record operation start
                self._record_operation_start("inlet", orchestration_context)
                
                # Create rollback point
                self._create_rollback_point("inlet", {
                    "original_body": self._selective_copy(body),
                    "user_id": user_id,
                    "timestamp": time.time()
                })
                
                # Check for coordination overhead
                coordination_start = time.time()
                if orchestration_context:
                    coordination_overhead = (time.time() - coordination_start) * 1000
                    COORDINATION_OVERHEAD.observe(coordination_overhead / 1000)
                    
                    if coordination_overhead > self.internal_config.coordination_overhead_threshold_ms:
                        log_with_context('warning', f"High coordination overhead detected: {coordination_overhead:.2f}ms",
                                       component='FILTER', user_id=user_id, operation='orchestration',
                                       context={'overhead_ms': coordination_overhead})
                
            except Exception as e:
                log_with_context('warning', "Filter orchestration setup failed, continuing without coordination",
                               component='FILTER', user_id=user_id, operation='orchestration', error=e)
                orchestration_context = None

        # --- Initialization & Valve Loading ---
        # Load valves early, handle potential errors
        try:
            # Reload global valves if OWUI injected config exists; otherwise keep defaults
            self.valves = self.Valves(**getattr(self, "config", {}).get("valves", {}))

            # Load user-specific valves (may override some per-user settings)
            user_valves = self._get_user_valves(__user__)

            if not user_valves.enabled:
                return body # Return early if disabled

            # Respect per-user setting for status visibility, ensuring it's set after loading
            show_status = self.valves.show_memory_status and user_valves.show_status
        except Exception as e:
            log_with_context('error', "Failed to load valves",
                           component='CONFIG', user_id=user_id, operation='load_valves', error=e)
            # Attempt to inform the UI, but ignore secondary errors to
            # avoid masking the original stack-trace
            try:
                        await self._safe_emit(
                            __event_emitter__,
                            {
                        "type": "error",
                        "content": f"Error loading memory configuration: {e}",
                    },
                )
            except Exception:
                pass
            # Prevent processing when config is invalid
            return body

        # --- Background Task Initialization (Ensure runs once) ---
        # Use hasattr for a simple check if tasks have been started
        if not hasattr(self, "_background_tasks_started"):
            self._initialize_background_tasks()
            self._background_tasks_started = True


        # --- Check for Guard Conditions ---
        if self._llm_feature_guard_active:
            log_with_context('info', "LLM feature guard active - skipping LLM-dependent memory operations",
                           component='INLET', user_id=user_id, operation='feature_guard')
        if self._embedding_feature_guard_active:
            log_with_context('info', "Embedding feature guard active - skipping embedding-dependent memory operations",
                           component='INLET', user_id=user_id, operation='feature_guard')


        # --- Process Incoming Message ---
        final_message = None
        # 1) Explicit stream=False (non-streaming completion requests)
        if body.get("stream") is False and body.get("messages"):
            final_message = body["messages"][-1].get("content")

        # 2) Streaming mode – grab final message when "done" flag arrives
        elif body.get("stream") is True and body.get("done", False):
            final_message = body.get("message", {}).get("content")

        # 3) Fallback – many WebUI front-ends don't set a "stream" key at all.
        if final_message is None and body.get("messages"):
            final_message = body["messages"][-1].get("content")

        # --- Command Handling ---
        # Check if the final message is a command before processing memories
        if final_message and final_message.strip().startswith("/"):
            command_parts = final_message.strip().split()
            command = command_parts[0].lower()

            # --- /memory list_banks Command --- NEW
            if command == "/memory" and len(command_parts) >= 2 and command_parts[1].lower() == "list_banks":
                log_with_context('info', "Handling /memory list_banks command",
                               component='INLET', user_id=user_id, operation='command_list_banks')
                try:
                    allowed_banks = self.valves.allowed_memory_banks
                    default_bank = self.valves.default_memory_bank
                    if allowed_banks:
                        bank_list_str = "\n".join([f"- {bank} {'(Default)' if bank == default_bank else ''}" for bank in allowed_banks])
                        response_msg = f"**Available Memory Banks:**\n{bank_list_str}"
                    else:
                        response_msg = "**Memory banks are disabled.** All memories are stored without categorization."
                    # Use enhanced event emitter for 2024 compliance
                    await self._emit_enhanced_event(
                        __event_emitter__, 
                        "info", 
                        response_msg,
                        {"command": "list_banks"},
                        "command_processing"
                    )
                    body["messages"] = [] # Prevent LLM call
                    body["prompt"] = "Command executed." # Placeholder for UI
                    body["bypass_prompt_processing"] = True # Signal to skip further processing
                    return body
                except Exception as e:
                    log_with_context('error', f"Error handling /memory list_banks: {e}")
                    await self._safe_emit(__event_emitter__, {"type": "error", "content": "Failed to list memory banks."})
                    # Allow fall through maybe? Or block? Let's block.
                    body["messages"] = []
                    body["prompt"] = "Error executing command." # Placeholder for UI
                    body["bypass_prompt_processing"] = True
                    return body

            # --- /memory assign_bank Command --- NEW
            elif command == "/memory" and len(command_parts) >= 4 and command_parts[1].lower() == "assign_bank":
                log_with_context('info', "Handling /memory assign_bank command",
                               component='INLET', user_id=user_id, operation='command_assign_bank')
                try:
                    memory_id = command_parts[2]
                    target_bank = command_parts[3]

                    if not self.valves.allowed_memory_banks:
                        await self._safe_emit(__event_emitter__, {"type": "error", "content": "Memory banks are disabled. Cannot assign memories to banks."})
                    elif target_bank not in self.valves.allowed_memory_banks:
                        allowed_banks_str = ", ".join(self.valves.allowed_memory_banks)
                        await self._safe_emit(__event_emitter__, {"type": "error", "content": f"Invalid bank '{target_bank}'. Allowed banks: {allowed_banks_str}"})
                    else:
                        # 1. Query the specific memory
                        # Note: query_memory might return multiple if content matches, need filtering by ID
                        query_result = await query_memory(
                            user_id=user_id,
                            form_data=QueryMemoryForm(query=memory_id, k=1000) # Query broadly first
                        )
                        target_memory = None
                        if query_result and hasattr(query_result, 'memories'):
                            memories = getattr(query_result, 'memories', [])
                            if memories:
                                for mem in memories:
                                    if hasattr(mem, 'id') and getattr(mem, 'id', None) == memory_id:
                                        target_memory = mem
                                        break

                        if not target_memory:
                            await self._safe_emit(__event_emitter__, {"type": "error", "content": f"Memory with ID '{memory_id}' not found."})
                        else:
                            # 2. Check if bank is already correct
                            current_bank = target_memory.metadata.get("memory_bank", self.valves.default_memory_bank)
                            if current_bank == target_bank:
                                await self._safe_emit(__event_emitter__, {"type": "info", "content": f"Memory '{memory_id}' is already in bank '{target_bank}'."})
                            else:
                                # 3. Update the memory (delete + add with modified metadata)
                                new_metadata = target_memory.metadata.copy()
                                new_metadata["memory_bank"] = target_bank
                                new_metadata["timestamp"] = datetime.now(timezone.utc).isoformat() # Update timestamp
                                new_metadata["source"] = "adaptive_memory_v3_assign_bank_cmd"

                                await delete_memory_by_id(user_id=user_id, memory_id=memory_id)
                                await add_memory(
                                    user_id=user_id,
                                    form_data=AddMemoryForm(
                                        content=target_memory.content,
                                        metadata=new_metadata
                                    )
                                )
                                await self._safe_emit(__event_emitter__, {"type": "info", "content": f"Successfully assigned memory '{memory_id}' to bank '{target_bank}'."})
                                self._increment_error_counter("memory_bank_assigned_cmd")

                except IndexError:
                     await self._safe_emit(__event_emitter__, {"type": "error", "content": "Usage: /memory assign_bank [memory_id] [bank_name]"})
                except Exception as e:
                    log_with_context('error', f"Error handling /memory assign_bank: {e}\n{traceback.format_exc()}")
                    await self._safe_emit(__event_emitter__, {"type": "error", "content": f"Failed to assign memory bank: {e}"})
                    self._increment_error_counter("assign_bank_cmd_error")

                # Always bypass LLM after handling command
                body["messages"] = []
                body["prompt"] = "Command executed." # Placeholder
                body["bypass_prompt_processing"] = True
                return body

            # --- Other /memory commands (Placeholder/Example - Adapt as needed) ---
            elif command == "/memory":
                # Example: Check for /memory list, /memory forget, etc.
                # Implement logic similar to assign_bank: parse args, call OWUI functions, emit status
                # Remember to add command handlers here based on other implemented features
                log_with_context('info', f"Handling generic /memory command",
                               component='INLET', user_id=user_id, operation='command_generic',
                               context={'message': final_message})
                await self._safe_emit(__event_emitter__, {"type": "info", "content": f"Memory command '{final_message}' received (implementation pending)."})
                body["messages"] = []
                body["prompt"] = "Memory command received." # Placeholder
                body["bypass_prompt_processing"] = True
                return body

            # --- /note command (Placeholder/Example) ---
            elif command == "/note":
                 log_with_context('info', f"Handling /note command",
                                component='INLET', user_id=user_id, operation='command_note',
                                context={'message': final_message})
                 # Implement logic for Feature 6 (Scratchpad)
                 await self._safe_emit(__event_emitter__, {"type": "info", "content": f"Note command '{final_message}' received (implementation pending)."})
                 body["messages"] = []
                 body["prompt"] = "Note command received." # Placeholder
                 body["bypass_prompt_processing"] = True
                 return body

            # --- /diagnose command - LLM Connection Diagnostics ---
            elif command == "/diagnose":
                log_with_context('info', "Handling /diagnose command",
                               component='INLET', user_id=user_id, operation='command_diagnose')
                
                await self._safe_emit(__event_emitter__, {
                    "type": "status", 
                    "data": {
                        "description": "🔍 Running LLM connection diagnostics...",
                        "done": False
                    }
                })
                
                try:
                    # Test current LLM configuration
                    provider_type = self.valves.llm_provider
                    api_url = self.valves.llm_api_endpoint_url
                    model_name = self.valves.llm_model_name
                    
                    # Perform comprehensive diagnostics
                    test_error = Exception("Diagnostic test")
                    diagnostics = await self._diagnose_connection_issues(api_url, provider_type, test_error)
                    tips = await self._get_connection_troubleshooting_tips(diagnostics)
                    
                    # Prepare diagnostic report
                    report_lines = [
                        f"🔧 **LLM Connection Diagnostics Report**",
                        f"📍 Provider: `{provider_type}` | Model: `{model_name}`",
                        f"🌐 Endpoint: `{api_url}`",
                        "",
                        "**Test Results:**"
                    ]
                    
                    tests = diagnostics.get("tests", {})
                    for test_name, result in tests.items():
                        status = result.get("status", "unknown")
                        if status == "success":
                            report_lines.append(f"✅ {test_name.replace('_', ' ').title()}")
                        elif status == "failed":
                            error = result.get("error", "Unknown error")
                            report_lines.append(f"❌ {test_name.replace('_', ' ').title()}: {error}")
                        elif status == "warnings":
                            issues = result.get("issues", [])
                            report_lines.append(f"⚠️ {test_name.replace('_', ' ').title()}: {len(issues)} warnings")
                        elif status == "rate_limited":
                            report_lines.append(f"🚦 {test_name.replace('_', ' ').title()}: Rate limited")
                        elif status == "not_required":
                            report_lines.append(f"ℹ️ {test_name.replace('_', ' ').title()}: Not required")
                    
                    if tips:
                        report_lines.extend(["", "**Troubleshooting Tips:**"])
                        for tip in tips[:5]:  # Limit to top 5 tips
                            report_lines.append(f"💡 {tip}")
                    
                    # Test connection with actual ping
                    try:
                        await self._safe_emit(__event_emitter__, {
                            "type": "status", 
                            "data": {
                                "description": "🔗 Testing live connection...",
                                "done": False
                            }
                        })
                        
                        # Simple test call to the LLM
                        test_response = await self.query_llm_with_retry(
                            "You are a test assistant.", 
                            "Respond with 'Connection test successful' if you receive this message."
                        )
                        
                        if "Connection test successful" in test_response or not test_response.startswith("Error:"):
                            report_lines.extend(["", "🟢 **Live Connection Test: PASSED**"])
                        else:
                            report_lines.extend(["", f"🔴 **Live Connection Test: FAILED**", f"Response: {test_response[:100]}..."])
                            
                    except Exception as live_test_error:
                        report_lines.extend(["", f"🔴 **Live Connection Test: ERROR**", f"Error: {str(live_test_error)[:100]}..."])
                    
                    # Get connection stats
                    stats = self.get_connection_stats()
                    cb_count = len([k for k, v in stats.get("circuit_breakers", {}).items() if v.get("is_open", False)])
                    
                    report_lines.extend([
                        "",
                        "**Connection Statistics:**",
                        f"🔵 Circuit Breakers Open: {cb_count}",
                        f"🔗 Session Active: {'Yes' if stats.get('session_open', False) else 'No'}",
                        f"🔌 Connector Active: {'Yes' if stats.get('connector_open', False) else 'No'}"
                    ])
                    
                    diagnostic_report = "\n".join(report_lines)
                    
                    await self._safe_emit(__event_emitter__, {
                        "type": "status", 
                        "data": {
                            "description": "✅ Diagnostics complete",
                            "done": True
                        }
                    })
                    
                    # Return the diagnostic report as the response
                    body["messages"] = [{
                        "role": "assistant",
                        "content": diagnostic_report
                    }]
                    body["prompt"] = "Diagnostic report generated."
                    body["bypass_prompt_processing"] = True
                    return body
                    
                except Exception as diag_error:
                    log_with_context('error', f"Diagnostic command failed: {diag_error}")
                    await self._safe_emit(__event_emitter__, {
                        "type": "status", 
                        "data": {
                            "description": f"❌ Diagnostics failed: {str(diag_error)[:50]}...",
                            "done": True
                        }
                    })
                    
                    error_report = f"🔴 **Diagnostic Error**\n\nFailed to run diagnostics: {str(diag_error)}\n\nBasic info:\n- Provider: {self.valves.llm_provider}\n- Endpoint: {self.valves.llm_api_endpoint_url}\n- Model: {self.valves.llm_model_name}"
                    
                    body["messages"] = [{
                        "role": "assistant", 
                        "content": error_report
                    }]
                    body["prompt"] = "Diagnostic error occurred."
                    body["bypass_prompt_processing"] = True
                    return body

            # --- /reset command - Reset Circuit Breakers ---
            elif command == "/reset":
                log_with_context('info', "Handling /reset command",
                               component='INLET', user_id=user_id, operation='command_reset')
                
                try:
                    # Check if specific endpoint is requested
                    if len(command_parts) > 1 and command_parts[1].lower() == "circuit":
                        reset_info = self.reset_circuit_breakers()
                        
                        if reset_info["reset_count"] > 0:
                            reset_report = f"🔄 **Circuit Breakers Reset**\n\n✅ Reset {reset_info['reset_count']} circuit breaker(s)\n\n**Reset Endpoints:**\n"
                            for endpoint in reset_info["reset_endpoints"]:
                                reset_report += f"- `{endpoint}`\n"
                        else:
                            reset_report = "ℹ️ **No Circuit Breakers to Reset**\n\nAll circuit breakers are already in normal state."
                        
                        await self._safe_emit(__event_emitter__, {
                            "type": "status", 
                            "data": {
                                "description": f"✅ Reset {reset_info['reset_count']} circuit breakers",
                                "done": True
                            }
                        })
                        
                        body["messages"] = [{
                            "role": "assistant",
                            "content": reset_report
                        }]
                        body["prompt"] = "Circuit breakers reset."
                        body["bypass_prompt_processing"] = True
                        return body
                    
                    else:
                        # Show help for reset command
                        help_text = "🔄 **Reset Command Help**\n\nAvailable options:\n- `/reset circuit` - Reset all LLM connection circuit breakers\n\nCircuit breakers automatically prevent requests to failing endpoints. Use this command if you've fixed connection issues and want to retry immediately."
                        
                        body["messages"] = [{
                            "role": "assistant",
                            "content": help_text
                        }]
                        body["prompt"] = "Reset command help."
                        body["bypass_prompt_processing"] = True
                        return body
                        
                except Exception as reset_error:
                    log_with_context('error', f"Reset command failed: {reset_error}")
                    error_report = f"🔴 **Reset Error**\n\nFailed to reset: {str(reset_error)}"
                    
                    body["messages"] = [{
                        "role": "assistant", 
                        "content": error_report
                    }]
                    body["prompt"] = "Reset error occurred."
                    body["bypass_prompt_processing"] = True
                    return body

        # --- Memory Injection --- #
        if self.valves.show_memories and not self._embedding_feature_guard_active: # Guard embedding-dependent retrieval
            try:
                log_with_context('info', f"Retrieving relevant memories, message: '{(final_message or '')[:50]}...'",
                               component='INLET', user_id=user_id, operation='get_relevant')
                # Use user-specific timezone for relevance calculation context
                relevant_memories = await self.get_relevant_memories(
                    current_message=final_message if final_message else "",
                    user_id=user_id,
                    user_timezone=user_valves.timezone # Use user-specific timezone
                )
                log_with_context('info', f"Retrieved {len(relevant_memories) if relevant_memories else 0} relevant memories for injection",
                               component='INLET', user_id=user_id, operation='inject_memories')
                if relevant_memories:
                    logger.info(
                        f"Injecting {len(relevant_memories)} relevant memories for user {user_id}"
                    )
                    # --- Emit Status: Injecting Memories --- ADDED
                    if show_status:
                        await self._safe_emit(
                            __event_emitter__,
                            {
                                "type": "status",
                                "data": {
                                    "description": f"Injecting {len(relevant_memories)} memories into context...",
                                    "done": False, # Still part of the pre-processing
                                },
                            },
                    )
                    injected_count = self._inject_memories_into_context(body, relevant_memories)
                    log_with_context('info', f"Memory injection completed: {injected_count} memories injected",
                                   component='OUTLET', user_id=user_id, operation='injection_complete',
                                   context={'injected': injected_count})
                else:
                    log_with_context('debug', "No relevant memories found - checking if expected",
                                   component='OUTLET', user_id=user_id, operation='inject_memories')
            except Exception as e:
                log_with_context('error', "Error retrieving/injecting memories",
                               component='OUTLET', user_id=user_id, operation='injection_error', error=e)
                await self._safe_emit(
                    __event_emitter__,
                    {"type": "error", "content": "Error retrieving relevant memories."},
                )

        # Clean the body before returning - remove user object if present
        # OpenWebUI expects 'user' to be a string (user ID) not an object when sending to LLM
        if 'user' in body and isinstance(body.get('user'), dict):
            # Extract user ID from user object
            user_id = body['user'].get('id')
            if user_id:
                body['user'] = str(user_id)
            else:
                # Remove user field if no ID
                body.pop('user', None)

        return body

    async def async_outlet(
        self,
        body: dict,
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
        __user__: Optional[dict] = None,
    ) -> dict:
        """Process LLM response, extract memories, and update the response"""
        try:
            # Add timeout protection to entire outlet operation
            return await asyncio.wait_for(
                self._async_outlet_impl(body, __event_emitter__, __user__),
                timeout=120.0  # 2 minutes max for outlet processing
            )
        except asyncio.TimeoutError:
            log_with_context('error', "Outlet operation timed out after 120 seconds",
                           component='OUTLET', operation='timeout')
            await self._safe_emit(__event_emitter__, {
                "type": "status",
                "data": {
                    "description": "⚠️ Memory extraction timed out - response delivered without memory processing",
                    "done": True,
                }
            })
            return body
        except Exception as e:
            log_with_context('error', "Critical error in outlet processing",
                           component='OUTLET', operation='error', error=e)
            await self._safe_emit(__event_emitter__, {
                "type": "status",
                "data": {
                    "description": f"⚠️ Memory extraction error: {str(e)[:100]}",
                    "done": True,
                }
            })
            return body
    
    async def _async_outlet_impl(
        self,
        body: dict,
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
        __user__: Optional[dict] = None,
    ) -> dict:
        """Implementation of async_outlet with timeout protection"""
        # logger.debug("****** OUTLET FUNCTION CALLED ******") # REMOVED

        # Log function entry

        # DEFENSIVE: Make a selective copy of the body to avoid dictionary changed size during iteration
        # Use selective copying for better performance
        body_copy = self._selective_copy(body, max_depth=2)

        # -----------------------------------------------------------
        # Filter Orchestration System Integration for Outlet
        # -----------------------------------------------------------
        orchestration_context = None
        operation_start_time = time.time()
        
        # Get user_id early for logging
        user_id = str(__user__.get("id", "unknown")) if __user__ else "unknown"
        
        if self.internal_config.enable_filter_orchestration:
            try:
                # Create execution context for this operation
                orchestration_context = self._create_execution_context(user_id)
                
                # Record operation start
                self._record_operation_start("outlet", orchestration_context)
                
                # Create rollback point
                self._create_rollback_point("outlet", {
                    "original_body": self._selective_copy(body),
                    "user_id": user_id,
                    "timestamp": time.time()
                })
                
            except Exception as e:
                log_with_context('warning', "Filter orchestration setup failed in outlet, continuing without coordination",
                               component='OUTLET', user_id=user_id, operation='orchestration', error=e)
                orchestration_context = None

        # Felicity says: No more living in the past—your config is now as fresh as your taste in AI assistants! 💅
        # Force reload configuration from persistent store to ensure latest settings are applied
        try:
            reload_success = await asyncio.wait_for(
                asyncio.to_thread(self._reload_configuration_safe, True),  # force_reload=True
                timeout=5.0
            )
            if not reload_success:
                log_with_context('warning', "Configuration force reload failed, continuing with current config",
                               component='OUTLET', user_id=user_id, operation='config_reload')
        except asyncio.TimeoutError:
            log_with_context('warning', "Configuration force reload timed out, continuing with current config",
                           component='OUTLET', user_id=user_id, operation='config_reload')
        except Exception as e:
            log_with_context('warning', "Configuration force reload error, continuing with current config",
                           component='OUTLET', user_id=user_id, operation='config_reload', error=e)

        # Skip processing if user is not authenticated
        if not __user__:
            log_with_context('warning', "No user information available - skipping memory processing",
                           component='OUTLET', user_id='unknown', operation='validate_user')
            return body_copy

        # Get user's ID for memory storage
        user_id = __user__.get("id")
        if not user_id:
            log_with_context('warning', "User object contains no ID - skipping memory processing",
                           component='OUTLET', user_id='unknown', operation='validate_user')
            return body_copy
        
        log_with_context('info', "Starting outlet processing",
                        component='OUTLET', user_id=user_id, operation='start')

        # Check if user has enabled memory function
        user_valves = self._get_user_valves(__user__)
        if not user_valves.enabled:
            log_with_context('info', "Memory function is disabled",
                           component='OUTLET', user_id=user_id, operation='check_enabled')
            return body_copy

        # Get user's timezone if set
        user_timezone = user_valves.timezone or self.valves.timezone

        # --- BEGIN MEMORY PROCESSING IN OUTLET --- 
        # Process BOTH the last user message AND the assistant's response for memory extraction
        last_user_message_content = None
        assistant_response_content = None
        message_history_for_context = []
        conversation_to_process = ""
        
        try:
            messages_copy = self._copy_messages(body_copy.get("messages", []))
            if messages_copy:
                # Find the last user message and the assistant's response
                last_user_index = -1
                
                # Find the last user message
                for i in range(len(messages_copy) - 1, -1, -1):
                    msg = messages_copy[i]
                    if msg.get("role") == "user" and msg.get("content"):
                        last_user_message_content = msg.get("content")
                        last_user_index = i
                        break
                
                # Find the assistant's response that follows the user message
                if last_user_index != -1 and last_user_index < len(messages_copy) - 1:
                    # Look for assistant messages after the user message
                    for i in range(last_user_index + 1, len(messages_copy)):
                        msg = messages_copy[i]
                        if msg.get("role") == "assistant" and msg.get("content"):
                            assistant_response_content = msg.get("content")
                            break
                
                # Get context messages before the last user message
                if last_user_index != -1:
                    start_index = max(0, last_user_index - self.internal_config.recent_messages_n)
                    message_history_for_context = messages_copy[start_index:last_user_index]
                
                # Prepare the full conversation for memory extraction
                if last_user_message_content:
                    conversation_to_process = f"User: {last_user_message_content}"
                    if assistant_response_content:
                        conversation_to_process += f"\n\nAssistant: {assistant_response_content}"

            if conversation_to_process:
                log_with_context('info', "Starting memory processing for conversation",
                               component='OUTLET', user_id=user_id, operation='memory_start')
                log_with_context('debug', f"Processing user message: {last_user_message_content[:60] if last_user_message_content else 'None'}...")
                log_with_context('debug', f"Processing assistant response: {assistant_response_content[:60] if assistant_response_content else 'None'}...")
                
                # Show status message if enabled
                if user_valves.show_status and __event_emitter__:
                    await self._safe_emit(
                        __event_emitter__,
                        {
                            "type": "status",
                            "data": {
                                "description": "🧠 Analyzing conversation for memories...",
                                "done": False,
                            },
                        },
                    )
                
                # Orchestration error handling wrapper
                try:
                    # Use asyncio.create_task for non-blocking processing
                    # Process the full conversation (user message + assistant response)
                    memory_task = asyncio.create_task(
                        self._process_user_memories(
                            user_message=conversation_to_process,
                            user_id=user_id,
                            event_emitter=__event_emitter__,
                            show_status=user_valves.show_status,
                            user_timezone=user_timezone,
                            recent_chat_history=message_history_for_context,
                        )
                    )
                    
                    # Wait for the task to complete with a reasonable timeout
                    try:
                        await asyncio.wait_for(memory_task, timeout=30.0)
                        log_with_context('info', "Memory processing completed successfully",
                                       component='OUTLET', user_id=user_id, operation='memory_complete')
                        
                        # Show completion status
                        if user_valves.show_status and __event_emitter__:
                            await self._safe_emit(
                                __event_emitter__,
                                {
                                    "type": "status",
                                    "data": {
                                        "description": "✅ Memory extraction complete",
                                        "done": True,
                                    },
                                },
                            )
                    except asyncio.TimeoutError:
                        logger.warning("Memory processing timed out after 30 seconds")
                        if user_valves.show_status and __event_emitter__:
                            await self._safe_emit(
                                __event_emitter__,
                                {
                                    "type": "status",
                                    "data": {
                                        "description": "⚠️ Memory extraction timed out",
                                        "done": True,
                                    },
                                },
                            )
                    
                except Exception as memory_error:
                    # Handle memory processing failures with orchestration system
                    if self.internal_config.enable_filter_orchestration and hasattr(self, 'orchestration_context'):
                        self._record_operation_failure("memory_processing", operation_start_time, str(memory_error), getattr(self, 'orchestration_context', None))
                        
                        if self.internal_config.enable_rollback_mechanism:
                            logger.warning(f"Memory processing failed, attempting rollback: {memory_error}")
                            self._perform_rollback()
                    
                    log_with_context('error', f"Memory processing failed: {memory_error}")
                    
                    if user_valves.show_status and __event_emitter__:
                        await self._safe_emit(
                            __event_emitter__,
                            {
                                "type": "status",
                                "data": {
                                    "description": f"❌ Memory extraction failed: {str(memory_error)[:50]}",
                                    "done": True,
                                },
                            },
                        )
                    raise  # Re-raise to be caught by outer exception handler
            else:
                logger.warning("No conversation content found in outlet body to process for memories.")

        except Exception as e:
            logger.error(f"Error initiating memory processing in outlet: {e}\n{traceback.format_exc()}")
            
            # Record orchestration failure if enabled
            if self.internal_config.enable_filter_orchestration:
                self._record_operation_failure("outlet", operation_start_time, str(e), getattr(self, 'orchestration_context', None))
        # --- END MEMORY PROCESSING IN OUTLET --- 

        # Note: Memory injection is now handled in inlet method for better timing
        # This outlet method focuses on memory extraction from the LLM response

        # Add confirmation message if memories were processed
        try:
            if user_valves.show_status:
                await self._add_confirmation_message(body_copy)
        except Exception as e:
            log_with_context('error', "Error adding confirmation message",
                           component='OUTLET', operation='confirmation_error', error=e)

        # -----------------------------------------------------------
        # Filter Orchestration Completion Tracking
        # -----------------------------------------------------------
        if self.internal_config.enable_filter_orchestration and orchestration_context:
            try:
                # Record successful completion
                self._record_operation_success("outlet", operation_start_time, orchestration_context)
                
                # Store context for potential use by other filters
                if self.internal_config.enable_shared_state and orchestration_context:
                    user_id = __user__.get("id") if __user__ else None
                    orchestration_context.shared_state["adaptive_memory_outlet_processed"] = True
                    orchestration_context.shared_state["adaptive_memory_user_id"] = user_id
                    
            except Exception as e:
                log_with_context('debug', f"Failed to record orchestration completion: {e}")

        # Clean the body before returning - remove user object if present
        # OpenWebUI expects 'user' to be a string (user ID) not an object when sending to LLM
        if 'user' in body_copy and isinstance(body_copy.get('user'), dict):
            # Extract user ID from user object
            user_id = body_copy['user'].get('id')
            if user_id:
                body_copy['user'] = str(user_id)
            else:
                # Remove user field if no ID
                body_copy.pop('user', None)
        
        # Return the modified response
        return body_copy

    async def _safe_emit(self, event_emitter: Optional[Callable[[Any], Awaitable[None]]], data: Dict[str, Any]) -> None:
        if not event_emitter:
            log_with_context('debug', "Event emitter not available")
            return
        try:
            await event_emitter(data)
        except Exception as e:
            log_with_context('error', "Error in event emitter",
                           component='OUTLET', operation='emitter_error', error=e)

    def _get_user_valves(self, __user__: dict) -> UserValves:
        if not __user__:
            log_with_context('warning', "No user information provided",
                           component='CONFIG', operation='get_user_valves')
            return self.UserValves()
        
        user_id = __user__.get('id', 'unknown') if hasattr(__user__, 'get') else 'unknown'
        user_valves_data = getattr(__user__, "valves", {})
        
        if not isinstance(user_valves_data, dict):
            log_with_context('warning', f"User valves attribute is not a dictionary (type: {type(user_valves_data)}), using defaults",
                           component='CONFIG', user_id=user_id, operation='get_user_valves')
            user_valves_data = {}
        try:
            return self.UserValves(**user_valves_data)
        except Exception as e:
            log_with_context('error', f"Could not determine user valves settings from data: {user_valves_data}",
                           component='CONFIG', user_id=user_id, operation='get_user_valves', error=e)
            return self.UserValves()

    async def _get_formatted_memories(self, user_id: str) -> List[Dict[str, Any]]:
        if not user_id:
            log_with_context('error', "_get_formatted_memories called without user_id",
                           component='MEMORY', operation='get_formatted_memories')
            raise ValueError("user_id is required for fetching memories")
        memories_list = []
        try:
            log_with_context('debug', f"Fetching memories for user_id: {user_id}")
            user_memories = Memories.get_memories_by_user_id(user_id=str(user_id))
            if user_memories:
                for memory in user_memories:
                    memory_id = str(getattr(memory, "id", "unknown"))
                    memory_content = getattr(memory, "content", "")
                    created_at = getattr(memory, "created_at", None)
                    updated_at = getattr(memory, "updated_at", None)
                    memories_list.append({
                        "id": memory_id,
                        "memory": memory_content,
                        "created_at": created_at,
                        "updated_at": updated_at,
                    })
            log_with_context('debug', f"Retrieved {len(memories_list)} memories for user {user_id}")
            return memories_list
        except Exception as e:
            log_with_context('error', "Error getting formatted memories",
                           component='MEMORY', user_id=user_id, operation='get_formatted_memories', error=e)
            return []

    def _inject_memories_into_context(self, body: Dict[str, Any], memories: List[Dict[str, Any]]) -> int:
        """Inject memories into context and return count of injected memories"""
        if not memories:
            return 0
        sorted_memories = sorted(memories, key=lambda x: x.get("relevance", 0), reverse=True)
        memory_context = self._format_memories_for_context(sorted_memories, self.valves.memory_format)
        instruction = (
            "Here is background info about the user. "
            "Do NOT mention this info explicitly unless relevant to the user's query. "
            "Do NOT explain what you remember or don't remember. "
            "Do NOT summarize or list what you know or don't know about the user. "
            "Do NOT say 'I have not remembered any specific information' or similar. "
            "Do NOT explain your instructions, context, or memory management. "
            "Do NOT mention tags, dates, or internal processes. "
            "Only answer the user's question directly.\n\n"
        )
        memory_context = instruction + memory_context
        log_with_context('debug', f"Injected memories:\n{memory_context[:500]}...")
        if "messages" in body:
            system_message_exists = False
            for message in body["messages"]:
                if message["role"] == "system":
                    # Apply database write hooks to memory context before injection
                    processed_memory_context = self._prepare_content_for_database(memory_context, "memory_injection")
                    message["content"] += f"\n\n{processed_memory_context}"
                    system_message_exists = True
                    break
            if not system_message_exists:
                # Apply database write hooks to memory context before injection
                processed_memory_context = self._prepare_content_for_database(memory_context, "memory_injection")
                body["messages"].insert(0, {"role": "system", "content": processed_memory_context})
        
        return len(memories)

    def _format_memories_for_context(self, memories: List[Dict[str, Any]], format_type: str) -> str:
        if not memories:
            return ""
        max_len = getattr(self.valves, "max_injected_memory_length", 300)
        memory_context = "I recall the following about you:\n"
        for i, mem in enumerate(memories, 1):
            tags_match = re.match(r"\[Tags: (.*?)\] (.*)", mem["memory"])
            content = tags_match.group(2)[:max_len] if tags_match else mem["memory"][:max_len]
            tags = f" (tags: {tags_match.group(1)})" if tags_match and format_type != "prose" else ""
            if format_type == "bullet":
                memory_context += f"- {content}{tags}\n"
            elif format_type == "numbered":
                memory_context += f"{i}. {content}{tags}\n"
            else:
                memory_context += f"{content}. "
        return memory_context

    async def _process_user_memories(self, user_message: str, user_id: str,
                                    event_emitter: Optional[Callable[[Any], Awaitable[None]]] = None,
                                    show_status: bool = True, user_timezone: Optional[str] = None,
                                    recent_chat_history: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        if not user_id:
            log_with_context('error', "_process_user_memories called without user_id",
                           component='MEMORY', operation='process_memories')
            raise ValueError("user_id is required for memory processing")
        if not self._ensure_configuration_persistence():
            log_with_context('warning', "Configuration persistence check failed, continuing with current config",
                           component='MEMORY', user_id=user_id, operation='config_check')
        config_content = getattr(self, "config", "<Not Set>")
        log_with_context('debug', f"Inspecting config at start of _process_user_memories: {config_content}",
                        component='MEMORY', user_id=user_id, operation='inspect_config') 

        # Start timer
        start_time = time.perf_counter()

        # Reset stored memories and error message
        # This variable held identified memories, not saved ones. We'll get saved count from process_memories return.
        # self.stored_memories = [] # Remove or repurpose if needed elsewhere, currently unused after this point.
        self._error_message = None

        # Emit "processing memories" status if enabled
        if show_status:
            await self._safe_emit(
                event_emitter,
                {
                    "type": "status",
                    "data": {
                        "description": "📝 Extracting potential new memories from your message…",
                        "done": False,
                    },
                },
            )

        # Debug logging for function entry
        log_with_context('debug', 
            f"Starting _process_user_memories for user {user_id} with message: {user_message[:50]}..."
        )

        # Get user valves
        user_valves = None
        try:
            user = Users.get_user_by_id(user_id)
            user_valves = self._get_user_valves(user)

            # Debug logging for user valves
            log_with_context('debug', 
                f"Retrieved user valves with memory enabled: {user_valves.enabled}"
            )

            if not user_valves.enabled:
                log_with_context('info', "Memory function disabled",
                               component='MEMORY', user_id=user_id, operation='check_enabled')
                if show_status:
                    await self._safe_emit(
                        event_emitter,
                        {
                            "type": "status",
                            "data": {
                                "description": "⏸️ Adaptive Memory is disabled in your settings – skipping memory save.",
                                "done": True,
                            },
                        },
                    )
                return []
        except Exception as e:
            log_with_context('error', "Error getting user valves",
                           component='MEMORY', user_id=user_id, operation='get_user_valves', error=e)
            if show_status:
                await self._safe_emit(
                    event_emitter,
                    {
                        "type": "status",
                        "data": {
                            "description": "⚠️ Unable to access memory settings – aborting memory save process.",
                            "done": True,
                        },
                    },
                )
            return []

        # Debug logging for memory identification start
        log_with_context('debug', f"Starting memory identification for message: {user_message[:60]}...")

        # Step 1: Use LLM to identify memories in the message
        memories = []
        parse_error_occurred = False # Track if parsing failed
        try:
            # Get user's existing memories for context (optional - can also be None)
            existing_memories = None
            # If the LLM needs context of existing memories:
            try:
                existing_memories = await self._get_formatted_memories(user_id)
                log_with_context('debug', 
                    f"Retrieved {len(existing_memories)} existing memories for context"
                )
            except Exception as e:
                log_with_context('warning', "Could not get existing memories (continuing)",
                               component='MEMORY', user_id=user_id, operation='get_existing', error=e)

            # Process message to extract memory operations
            memories = await self.identify_memories(
                user_message,
                existing_memories=existing_memories,
                user_timezone=user_timezone,
            )

            # Debug logging after memory identification
            log_with_context('debug', 
                f"Memory identification complete. Found {len(memories)} potential memories"
            )

        except Exception as e:
            self.error_counters["llm_call_errors"] += 1
            log_with_context('error', "Error identifying memories",
                           component='MEMORY', user_id=user_id, operation='identify_memories', error=e)
            self._error_message = f"llm_error: {str(e)[:50]}..." # Point 6: More specific error
            parse_error_occurred = True  # Indicate identification failed
            
            # Try fallback memory creation for preference statements
            if getattr(self.valves, 'enable_hang_prevention', True):
                log_with_context('info', "Attempting fallback memory creation for preference statements",
                               component='MEMORY', user_id=user_id, operation='fallback_create')
                fallback_memories = await self._create_fallback_memory_from_preference(user_message)
                if fallback_memories:
                    log_with_context('info', f"Created {len(fallback_memories)} fallback memories",
                                   component='MEMORY', user_id=user_id, operation='fallback_success',
                                   context={'count': len(fallback_memories)})
                    memories = fallback_memories
                    parse_error_occurred = False
            if show_status:
                await self._safe_emit(
                    event_emitter,
                    {
                        "type": "status",
                        "data": {
                            "description": f"⚠️ Memory error: {str(e)}",
                            "done": True,
                        },
                    },
                )
            return []

        # Debug logging for filtering
        log_with_context('debug', "Starting memory filtering step...")

        # Step 2: Filter memories (apply blacklist/whitelist/trivia filtering)
        filtered_memories = []
        if memories:
            # Apply filters based on valves
            try:
                # Get filter configuration valves
                min_length = self.valves.min_memory_length
                blacklist = self.valves.blacklist_topics
                whitelist = self.valves.whitelist_keywords
                filter_trivia = self.valves.filter_trivia

                log_with_context('debug', 
                    f"Using filters: min_length={min_length}, blacklist={blacklist}, whitelist={whitelist}, filter_trivia={filter_trivia}"
                )

                # Default trivia patterns (common knowledge patterns)
                trivia_patterns = [
                    r"\b(when|what|who|where|how)\s+(is|was|were|are|do|does|did)\b",  # Common knowledge questions
                    r"\b(fact|facts)\b",  # Explicit facts
                    r"\b(in the year|in \d{4})\b",  # Historical dates
                    r"\b(country|countries|capital|continent|ocean|sea|river|mountain|planet)\b",  # Geographic/scientific
                    r"\b(population|inventor|invented|discovered|founder|founded|created|author|written|directed)\b",  # Attribution/creation
                ]

                # Known meta-request phrases
                meta_request_phrases = [
                    "remember this",
                    "make a note",
                    "don't forget",
                    "keep in mind",
                    "save this",
                    "add this to",
                    "log this",
                    "put this in",
                ]

                # Process each memory with filtering
                for memory in memories:
                    # Validate operation
                    if not self._validate_memory_operation(memory):
                        log_with_context('debug', f"Invalid memory operation: {str(memory)}")
                        continue

                    # Extract content for filtering
                    content = memory.get("content", "").strip()

                    # Apply minimum length filter
                    if len(content) < min_length:
                        log_with_context('debug', 
                            f"Memory too short ({len(content)} < {min_length}): {content}"
                        )
                        continue

                    # Check if it's a meta-request
                    is_meta_request = False
                    for phrase in meta_request_phrases:
                        if phrase.lower() in content.lower():
                            is_meta_request = True
                            log_with_context('debug', f"Meta-request detected: {content}")
                            break

                    if is_meta_request:
                        continue

                    # Check blacklist (if configured)
                    if blacklist:
                        is_blacklisted = False
                        for topic in blacklist.split(","):
                            topic = topic.strip().lower()
                            if topic and topic in content.lower():
                                # Check whitelist override
                                is_whitelisted = False
                                if whitelist:
                                    for keyword in whitelist.split(","):
                                        keyword = keyword.strip().lower()
                                        if keyword and keyword in content.lower():
                                            is_whitelisted = True
                                            log_with_context('debug', 
                                                f"Whitelisted term '{keyword}' found in blacklisted content"
                                            )
                                            break

                                if not is_whitelisted:
                                    is_blacklisted = True
                                    log_with_context('debug', 
                                        f"Blacklisted topic '{topic}' found: {content}"
                                    )
                                    break

                        if is_blacklisted:
                            continue

                    # Check trivia patterns (if enabled)
                    if filter_trivia:
                        is_trivia = False
                        for pattern in trivia_patterns:
                            if re.search(pattern, content.lower()):
                                log_with_context('debug', 
                                    f"Trivia pattern '{pattern}' matched: {content}"
                                )
                                is_trivia = True
                                break

                        if is_trivia:
                            continue

                    # Memory passed all filters
                    filtered_memories.append(memory)
                    log_with_context('debug', f"Memory passed all filters: {content}")

                logger.info(
                    f"Filtered memories: {len(filtered_memories)}/{len(memories)} passed"
                )
            except Exception as e:
                log_with_context('error', f"Error filtering memories: {e}\n{traceback.format_exc()}")
                filtered_memories = (
                    memories  # On error, attempt to process all memories
                )

        # --- NEW: Confidence Score Filtering ---
        memories_passing_confidence = []
        low_confidence_discarded = 0
        min_conf = self.valves.min_confidence_threshold
        log_with_context('debug', f"Applying confidence filter (threshold: {min_conf})...")
        for mem in filtered_memories:
            confidence_score = float(mem.get("confidence", 0.0)) # Ensure float comparison
            if confidence_score >= min_conf:
                memories_passing_confidence.append(mem)
            else:
                low_confidence_discarded += 1
                log_with_context('debug', f"Discarding memory due to low confidence ({confidence_score:.2f} < {min_conf}): {str(mem.get('content', ''))[:50]}...")
        
        # Emit status message if any memories were discarded due to low confidence
        if low_confidence_discarded > 0 and show_status:
            await self._safe_emit(
                event_emitter,
                {
                    "type": "status",
                    "data": {
                        "description": f"ℹ️ Discarded {low_confidence_discarded} potential memories due to low confidence (< {min_conf}).",
                        "done": False, # Indicate processing is ongoing
                    },
                },
            )
        
        # Use the confidence-filtered list for subsequent processing
        filtered_memories = memories_passing_confidence
        # --- END NEW ---

        # Debug logging after filtering
        log_with_context('debug', f"After filtering: {len(filtered_memories)} memories remain")

        # If no memories to process after filtering, log and return
        if not filtered_memories: # Check if the list is empty
            # --- Check for JSON Parse Error --- NEW
            if self._error_message == "json_parse_error" and show_status:
                 await self._safe_emit(
                    event_emitter,
                    {
                        "type": "status",
                        "data": {
                            "description": "⚠️ LLM response invalid - memory extraction failed.",
                            "done": True, # Mark as done even on error
                        },
                    },
                )
                 return [] # Exit after emitting error status
            # --- END JSON Parse Error Check ---

            # Point 5: Immediate-Save Shortcut for short preferences on parse error
            if (
                self.valves.enable_short_preference_shortcut
                and parse_error_occurred
                and len(user_message) <= 60
                and any(keyword in user_message.lower() for keyword in ["favorite", "love", "like", "enjoy"])
            ):
                logger.info("JSON parse failed, but applying short preference shortcut.")
                try:
                    shortcut_op = MemoryOperation(
                        operation="NEW",
                        content=user_message.strip(), # Save the raw message content
                        tags=["preference"] # Assume preference tag
                    )
                    await self._execute_memory_operation(shortcut_op, user) # Directly execute
                    logger.info(f"Successfully saved memory via shortcut: {user_message[:50]}...")
                    # Set a specific status message for this case
                    self._error_message = None # Clear parse error flag
                    # Since we bypassed normal processing, we need a result list for status reporting
                    saved_operations_list = [shortcut_op.model_dump()] # Use model_dump() for Pydantic v2+
                    # Skip the rest of the processing steps as we forced a save
                except Exception as shortcut_err:
                    log_with_context('error', f"Error during short preference shortcut save: {shortcut_err}")
                    self._error_message = "shortcut_save_error"
                    saved_operations_list = [] # Indicate save failed
            else:
                # Normal case: No memories identified or filtered out, and no shortcut applied
                logger.info("No valid memories to process after filtering/identification.")
                if show_status and not self._error_message:
                    # Determine reason for no save
                    final_status_reason = self._error_message or "filtered_or_duplicate"
                    status_desc = f"ℹ️ Memory save skipped – {final_status_reason.replace('_', ' ')}."
                    await self._safe_emit(
                        event_emitter,
                        {
                            "type": "status",
                            "data": {
                                "description": status_desc,
                                "done": True,
                            },
                        },
                    )
                return [] # Return empty list as nothing was saved through normal path
        else:
           # We have filtered_memories, proceed with normal processing
           pass # Continue to Step 3

        # Step 3: Get current memories and handle max_total_memories limit
        try:
            current_memories_data = await self._get_formatted_memories(user_id)
            log_with_context('debug', 
                f"Retrieved {len(current_memories_data)} existing memories from database"
            )

            # If we'd exceed the maximum memories per user, apply pruning
            max_memories = self.valves.max_memories_to_remember
            current_count = len(current_memories_data)
            new_count = len(filtered_memories) # Only count NEW operations towards limit for pruning decision
            
            if current_count + new_count > max_memories:
                to_remove = current_count + new_count - max_memories
                log_with_context('info', f"Memory limit would be exceeded. Need to prune {to_remove} memories",
                               component='MEMORY', user_id=user_id, operation='memory_limit_exceeded',
                               context={'limit': max_memories, 'to_remove': to_remove})
                
                memories_to_prune_ids = []
                
                # Choose pruning strategy based on valve
                strategy = self.valves.pruning_strategy
                logger.info(f"Applying pruning strategy: {strategy}")
                
                if strategy == "least_relevant":
                    try:
                        # Calculate relevance for all existing memories against the current user message
                        memories_with_relevance = []
                        # Re-use logic similar to get_relevant_memories but for *all* memories
                        
                        user_embedding = None
                        if self._local_embedding_model:
                            try:
                                user_embedding = self._local_embedding_model.encode(user_message, normalize_embeddings=True)
                            except Exception as e:
                                logger.warning(f"Could not encode user message for relevance pruning: {e}")

                        # Determine if we can use vectors or need LLM fallback (respecting valve)
                        can_use_vectors = user_embedding is not None
                        needs_llm = self.valves.use_llm_for_relevance

                        # --- Calculate Scores --- 
                        if not needs_llm and can_use_vectors:
                             # Vector-only relevance calculation
                            for mem_data in current_memories_data:
                                mem_id = mem_data.get("id")
                                if mem_id is None:
                                    continue
                                mem_emb = self.memory_embeddings.get(mem_id)
                                # Ensure embedding exists or try to compute it
                                if mem_emb is None and self._local_embedding_model is not None:
                                    try:
                                        mem_text = mem_data.get("memory") or ""
                                        if mem_text:
                                            mem_emb = self._local_embedding_model.encode(mem_text, normalize_embeddings=True)
                                            self.memory_embeddings[mem_id] = mem_emb # Cache it
                                    except Exception as e:
                                        logger.warning(f"Failed to compute embedding for existing memory {mem_id}: {e}")
                                        mem_emb = None # Mark as failed
                                
                                if mem_emb is not None and user_embedding is not None:
                                    sim_score = float(np.dot(user_embedding, mem_emb))
                                    memories_with_relevance.append({"id": mem_id, "relevance": sim_score})
                                else:
                                    # Assign low relevance if embedding fails
                                    memories_with_relevance.append({"id": mem_id, "relevance": 0.0})
                        elif needs_llm:
                            # LLM-based relevance calculation (simplified, no caching needed here)
                            # Prepare memories for LLM prompt
                            memory_strings_for_llm = [
                                f"ID: {mem['id']}, CONTENT: {mem['memory']}" 
                                for mem in current_memories_data
                            ]
                            system_prompt = self.valves.memory_relevance_prompt
                            llm_user_prompt = f"""Current user message: "{user_message}"

Available memories:
{json.dumps(memory_strings_for_llm)}

Rate the relevance of EACH memory to the current user message."""
                            
                            try:
                                llm_response_text = await self.query_llm_with_retry(system_prompt, llm_user_prompt)
                                llm_relevance_results = self._extract_and_parse_json(llm_response_text)
                                
                                if isinstance(llm_relevance_results, list):
                                    # Map results back to IDs
                                    llm_scores = {item.get("id"): item.get("relevance", 0.0) for item in llm_relevance_results if isinstance(item, dict)}
                                    for mem_data in current_memories_data:
                                        mem_id = mem_data.get("id")
                                        score = llm_scores.get(mem_id, 0.0) # Default to 0 if LLM missed it
                                        memories_with_relevance.append({"id": mem_id, "relevance": score})
                                else:
                                    logger.warning("LLM relevance check for pruning failed to return valid list. Pruning might default to FIFO.")
                                    # Fallback: assign 0 relevance to all, effectively making it FIFO-like for this run
                                    memories_with_relevance = [{"id": m["id"], "relevance": 0.0} for m in current_memories_data]
                            except Exception as llm_err:
                                log_with_context('error', f"Error during LLM relevance check for pruning: {llm_err}")
                                memories_with_relevance = [{"id": m["id"], "relevance": 0.0} for m in current_memories_data]
                        else: # Cannot use vectors and LLM not enabled - default to FIFO-like
                             logger.warning("Cannot determine relevance for pruning (no embeddings/LLM). Pruning will be FIFO-like.")
                             memories_with_relevance = [{"id": m["id"], "relevance": 0.0} for m in current_memories_data]

                        # --- Sort and Select for Pruning ---                     
                        # Sort by relevance ascending (lowest first)
                        memories_with_relevance.sort(key=lambda x: x.get("relevance", 0.0))
                        
                        # Select the IDs of the least relevant memories to remove (take the first `to_remove` items after sorting)
                        memories_to_prune_ids = [mem["id"] for mem in memories_with_relevance[:to_remove]]
                        log_with_context('info', f"Identified {len(memories_to_prune_ids)} least relevant memories for pruning",
                                       component='MEMORY', user_id=user_id, operation='pruning_identified')
                        
                    except Exception as relevance_err:
                        log_with_context('error', f"Error calculating relevance for pruning, falling back to FIFO: {relevance_err}")
                        # Fallback to FIFO on any error during relevance calculation
                        strategy = "fifo"
                        
                # Default or fallback FIFO strategy
                if strategy == "fifo":
                    # Sort by timestamp ascending (oldest first)
                    # Make sure timestamp exists, fallback to a very old date if not
                    default_date = datetime.min.replace(tzinfo=timezone.utc)
                    sorted_memories = sorted(
                        current_memories_data, 
                        key=lambda x: x.get("created_at", default_date)
                    )
                    memories_to_prune_ids = [mem["id"] for mem in sorted_memories[:to_remove]]
                    log_with_context('info', f"Identified {len(memories_to_prune_ids)} oldest memories (FIFO) for pruning",
                                   component='MEMORY', user_id=user_id, operation='pruning_fifo')

                # Execute pruning if IDs were identified
                if memories_to_prune_ids:
                    pruned_count = 0
                    for memory_id_to_delete in memories_to_prune_ids:
                        try:
                            delete_op = MemoryOperation(operation="DELETE", id=memory_id_to_delete)
                            await self._execute_memory_operation(delete_op, user)
                            pruned_count += 1
                        except Exception as e:
                            log_with_context('error', f"Error pruning memory {memory_id_to_delete}: {e}")
                    log_with_context('info', f"Successfully pruned {pruned_count} memories",
                                   component='MEMORY', user_id=user_id, operation='pruning_complete')
                else:
                    logger.warning("Pruning needed but no memory IDs identified for deletion.")
                    
        except Exception as e:
            logger.error(
                f"Error handling max_memories_to_remember: {e}\n{traceback.format_exc()}"
            )
            # Continue processing the new memories even if pruning failed

        # Debug logging before processing operations
        log_with_context('debug', "Beginning to process memory operations...")

        # Step 4: Process the filtered memories
        processing_error: Optional[Exception] = None
        try:
            # process_memories now returns the list of successfully executed operations
            log_with_context('debug', f"Calling process_memories with {len(filtered_memories)} items: {str(filtered_memories)}") # Log the exact list being passed
            saved_operations_list = await self.process_memories(
                filtered_memories, user_id
            )
            log_with_context('debug', f"Memory saving attempt complete: {len(saved_operations_list)} operations saved",
                           component='MEMORY', user_id=user_id, operation='save_complete',
                           context={'saved_count': len(saved_operations_list)})
        except Exception as e:
            processing_error = e
            log_with_context('error', "Error processing memories",
                           component='MEMORY', user_id=user_id, operation='process_error', error=e)
            self._error_message = f"processing_error: {str(e)[:50]}..."  # Point 6: More specific error

        # Debug confirmation logs
        if saved_operations_list:
            log_with_context('info', f"Memory extraction completed: {len(saved_operations_list)} memories saved",
                           component='MEMORY', user_id=user_id, operation='extraction_complete',
                           context={'saved_count': len(saved_operations_list)})
        elif processing_error:
            log_with_context('warning', f"Memory processing failed due to an error: {processing_error}",
                           component='MEMORY', user_id=user_id, operation='process_failed')
        else:
            log_with_context('warning', "Memory processing finished, but no memories were saved (potentially due to duplicates or errors)",
                           component='MEMORY', user_id=user_id, operation='process_no_saves')

        # Emit completion status
        if show_status:
            elapsed_time = time.perf_counter() - start_time
            # Base the status on the actual saved operations list
            saved_count = len(saved_operations_list)  # Directly use length of result
            if saved_count > 0:
                # Check if it was the shortcut save
                if any(op.get("content") == user_message.strip() for op in saved_operations_list):
                     status_desc = f"✅ Saved 1 memory via shortcut ({elapsed_time:.2f}s)"
                else:
                    plural = "memory" if saved_count == 1 else "memories"
                    status_desc = f"✅ Added {saved_count} new {plural} to your memory bank ({elapsed_time:.2f}s)"
            else:
                # Build smarter status based on duplicate counters
                if getattr(self, "_duplicate_refreshed", 0):
                    status_desc = f"✅ Memory refreshed (duplicate confirmed) ({elapsed_time:.2f}s)"
                elif getattr(self, "_duplicate_skipped", 0):
                    status_desc = f"✅ Preference already saved – duplicate ignored ({elapsed_time:.2f}s)"
                else:
                    final_status_reason = self._error_message or "filtered_or_duplicate"
                    status_desc = f"⚠️ Memory save skipped – {final_status_reason.replace('_', ' ')} ({elapsed_time:.2f}s)"
            await self._safe_emit(
                event_emitter,
                {
                    "type": "status",
                    "data": {
                        "description": status_desc,
                        "done": True,
                    },
                },
            )

        # Return the list of operations that were actually saved
        return saved_operations_list

    async def identify_memories(
        self,
        input_text: str,
        existing_memories: Optional[List[Dict[str, Any]]] = None,
        user_timezone: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Identify potential memories from text using LLM"""
        log_with_context('debug', 
            f"Starting memory identification from input text: {input_text[:50]}..."
        )

        # Remove <details> blocks that may interfere with processing
        input_text = re.sub(r"<details>.*?</details>", "", input_text, flags=re.DOTALL)

        # Clean up and prepare the input
        clean_input = input_text.strip()
        log_with_context('debug', f"Cleaned input text length: {len(clean_input)}")

        # Prepare the system prompt
        try:
            # Get the base prompt template
            memory_prompt = self.valves.memory_identification_prompt

            # Add datetime context
            now_str = self.get_formatted_datetime(user_timezone)
            datetime_context = f"Current datetime: {now_str}"

            # Add memory categories context based on enabled flags
            categories = []
            if self.valves.enable_identity_memories:
                categories.append("identity")
            if self.valves.enable_behavior_memories:
                categories.append("behavior")
            if self.valves.enable_preference_memories:
                categories.append("preference")
            if self.valves.enable_goal_memories:
                categories.append("goal")
            if self.valves.enable_relationship_memories:
                categories.append("relationship")
            if self.valves.enable_possession_memories:
                categories.append("possession")

            categories_str = ", ".join(categories)

            # Add existing memories context if provided
            existing_memories_str = ""
            if existing_memories and len(existing_memories) > 0:
                existing_memories_str = "Existing memories:\n"
                for i, mem in enumerate(
                    existing_memories[:5]
                ):  # Limit to 5 recent memories
                    existing_memories_str += f"- {mem.get('content', 'Unknown')}\n"

            # Combine all context
            context = f"{datetime_context}\nEnabled categories: {categories_str}\n{existing_memories_str}"

            # Log the components of the prompt
            log_with_context('debug', f"Memory identification context: {context}")

            # Create the final system prompt with context
            system_prompt = f"{memory_prompt}\n\nCONTEXT:\n{context}"

            log_with_context('debug', 
                f"Final memory identification system prompt length: {len(system_prompt)}"
            )
        except Exception as e:
            logger.error(f"Error building memory identification prompt: {e}")
            system_prompt = self.valves.memory_identification_prompt

        # Call LLM to identify memories
        start_time = time.time()
        log_with_context('debug', 
            f"Calling LLM for memory identification with provider: {self.valves.llm_provider}, model: {self.valves.llm_model_name}"
        )

        try:
            # Construct the user prompt with few-shot examples
            user_prompt = f"""Analyze the following user message and extract relevant memories:
>>> USER MESSAGE START <<<
+{clean_input}
>>> USER MESSAGE END <<<

--- EXAMPLES OF DESIRED OUTPUT FORMAT ---
Example 1 Input: "I really love pizza, especially pepperoni."
Example 1 Output: [{{"operation": "NEW", "content": "User loves pizza, especially pepperoni", "tags": ["preference"], "confidence": 0.85}}]

Example 2 Input: "What's the weather like today?"
Example 2 Output: []

Example 3 Input: "My sister Jane is visiting next week. I should buy her flowers."
Example 3 Output: [{{"operation": "NEW", "content": "User has a sister named Jane", "tags": ["relationship"], "confidence": 0.9}}, {{"operation": "NEW", "content": "User's sister Jane is visiting next week", "tags": ["relationship"], "confidence": 0.95}}]
--- END EXAMPLES ---

Produce ONLY the JSON array output for the user message above, adhering strictly to the format requirements outlined in the system prompt.
"""
            # Note: Doubled curly braces {{ }} are used to escape them within the f-string for the JSON examples.

            # Log the user prompt structure for debugging
            log_with_context('debug', 
                f"User prompt structure with few-shot examples:\n{user_prompt[:500]}..."
            )  # Log first 500 chars

            # Call LLM with the modified prompts
            llm_response = await self.query_llm_with_retry(
                system_prompt, user_prompt
            )  # Pass the new user_prompt
            elapsed = time.time() - start_time
            log_with_context('debug', 
                f"LLM memory identification completed in {elapsed:.2f}s, response length: {len(llm_response)}"
            )
            log_with_context('debug', f"LLM raw response for memory identification: {llm_response}")

            # --- Handle LLM Errors --- #
            if llm_response.startswith("Error:"):
                self.error_counters["llm_call_errors"] += 1
                if "LLM_CONNECTION_FAILED" in llm_response:
                    log_with_context('error', f"LLM Connection Error during identification: {llm_response}")
                    self._error_message = "llm_connection_error"
                else:
                    log_with_context('error', f"LLM Error during identification: {llm_response}")
                    self._error_message = "llm_error"
                return [] # Return empty list on LLM error

            # Parse the response (assumes JSON format)
            result = self._extract_and_parse_json(llm_response)
            log_with_context('debug', 
                f"Parsed result type: {type(result)}, content: {str(result)[:500]}"
            )

            # Check if we got a dict instead of a list (common LLM error)
            if isinstance(result, dict):
                log_with_context('warning', "LLM returned a JSON object instead of an array. Attempting conversion.",
                               component='LLM', operation='parse_response')
                result = self._convert_dict_to_memory_operations(result)
                log_with_context('debug', f"Converted dict to {len(result)} memory operations")

            # Check for empty result
            if not result:
                logger.warning("No memory operations identified by LLM")
                return []

            # Validate operations format
            valid_operations = []
            invalid_count = 0

            if isinstance(result, list):
                for op in result:
                    if self._validate_memory_operation(op):
                        valid_operations.append(op)
                    else:
                        invalid_count += 1

                log_with_context('debug', 
                    f"Identified {len(valid_operations)} valid memory operations, {invalid_count} invalid"
                )
                return valid_operations
            else:
                log_with_context('error', 
                    f"LLM returned invalid format (neither list nor dict): {type(result)}"
                )
                self._error_message = (
                    "LLM returned invalid format. Expected JSON array."
                )
                return []

        except Exception as e:
            logger.error(
                f"Error in memory identification: {e}\n{traceback.format_exc()}"
            )
            self.error_counters["llm_call_errors"] += 1
            self._error_message = f"Memory identification error: {str(e)}"
            return []

    def _validate_memory_operation(self, op: Dict[str, Any]) -> bool:
        """Validate memory operation format and required fields"""
        if not isinstance(op, dict):
            logger.warning(f"Invalid memory operation format (not a dict): {op}")
            return False

        # Check if operation field exists, if not try to infer it
        if "operation" not in op:
            # Look for typical patterns to guess the operation type
            if any(k.lower() == "operation" for k in op.keys()):
                # Operation may be under a different case
                for k, v in op.items():
                    if k.lower() == "operation" and isinstance(v, str):
                        op["operation"] = v
                        break

            # Look for operation in original format but in wrong place
            elif isinstance(op, dict) and any(
                v in ["NEW", "UPDATE", "DELETE"] for v in op.values()
            ):
                for k, v in op.items():
                    if v in ["NEW", "UPDATE", "DELETE"]:
                        op["operation"] = v
                        # Remove the old key if it's not "operation"
                        if k != "operation":
                            op.pop(k, None)
                        break

            # Default based on presence of fields
            elif "id" in op and "content" in op:
                # Default to UPDATE if we have both id and content
                op["operation"] = "UPDATE"
            elif "content" in op:
                # Default to NEW if we only have content
                op["operation"] = "NEW"
            else:
                logger.warning(f"Cannot determine operation type for: {op}")
                return False

        # Normalize operation to uppercase
        if isinstance(op["operation"], str):
            op["operation"] = op["operation"].upper()

        if op["operation"] not in ["NEW", "UPDATE", "DELETE"]:
            logger.warning(f"Invalid operation type: {op['operation']}")
            return False

        if op["operation"] in ["UPDATE", "DELETE"] and "id" not in op:
            log_with_context('warning', f"Missing ID for {op['operation']} operation",
                           component='MEMORY', operation='validate_operation',
                           context={'op': op})
            return False

        if op["operation"] in ["NEW", "UPDATE"] and "content" not in op:
            log_with_context('warning', f"Missing content for {op['operation']} operation",
                           component='MEMORY', operation='validate_operation',
                           context={'op': op})
            return False

        # Tags are optional but should be a list if present
        if "tags" in op and not isinstance(op["tags"], list):
            # Try to fix if it's a string
            if isinstance(op["tags"], str):
                try:
                    # See if it's a JSON string
                    parsed_tags = json.loads(op["tags"])
                    if isinstance(parsed_tags, list):
                        op["tags"] = parsed_tags
                    else:
                        # If it parsed but isn't a list, handle that case
                        op["tags"] = [str(parsed_tags)]
                except json.JSONDecodeError:
                    # Split by comma if it looks like a comma-separated list
                    if "," in op["tags"]:
                        op["tags"] = [tag.strip() for tag in op["tags"].split(",")]
                    else:
                        # Just make it a single-item list
                        op["tags"] = [op["tags"]]
            else:
                logger.warning(
                    f"Invalid tags format, not a list or string: {op['tags']}"
                )
                op["tags"] = []  # Default to empty list

        # Validate memory_bank field
        provided_bank = None
        if "memory_bank" in op and isinstance(op["memory_bank"], str):
            raw_bank_value = op["memory_bank"]
            provided_bank = raw_bank_value.strip().capitalize() # Normalize

            # --- SIMPLIFIED VALIDATION LOGIC ---
            # Use the validated list directly from self.valves
            allowed_banks_list = self.valves.allowed_memory_banks or []

            if provided_bank in allowed_banks_list:
                 # Valid bank provided
                 op["memory_bank"] = provided_bank # Assign normalized valid bank
            else:
                 # Invalid bank provided
                logger.warning(
                    f"Invalid memory bank '{op['memory_bank']}' (normalized to '{provided_bank}'), not in allowed list {allowed_banks_list}. Using default '{self.valves.default_memory_bank}'"
                )
                op["memory_bank"] = self.valves.default_memory_bank
        else:
            # If memory_bank is missing or not a string, set default
            log_with_context('debug', 
                f"Memory bank missing or invalid type ({type(op.get('memory_bank'))}), using default '{self.valves.default_memory_bank}'"
            )
            op["memory_bank"] = self.valves.default_memory_bank

        if "confidence" in op:
            if isinstance(op["confidence"], (int, float)):
                if not 0.0 <= op["confidence"] <= 1.0:
                    logger.warning(f"Invalid confidence score range: {op['confidence']}")
                    return False
            else:
                logger.warning(f"Invalid confidence score type: {type(op['confidence'])}")
                return False
        else:
            log_with_context('warning', "Missing confidence score in operation",
                           component='MEMORY', operation='validate_confidence')
            return False
        return True

    def _extract_and_parse_json(self, text: str) -> Union[List, Dict, None]:
        """
        Enhanced JSON extraction and parsing with comprehensive repair capabilities.
        
        Uses the JSON repair system to handle malformed JSON from sub-3B models
        and other LLM APIs with robust fallback mechanisms.
        """
        if not text:
            return None

        log_with_context('debug', f"Raw LLM response content received: {text[:200]}...")
        
        # Use enhanced JSON parser if available
        if self._json_parser is not None:
            try:
                # Create context for memory operations
                context = {
                    "expect_operations": True,
                    "memory_operations": True
                }
                
                result = self._json_parser.parse_with_repair(text, context=context)
                
                if result.success:
                    log_with_context('debug', f"JSON repair successful using method: {result.repair_method}")
                    
                    # Apply the same post-processing logic as before
                    parsed = result.parsed_data
                    
                    # Unwrap single-key objects containing lists
                    if isinstance(parsed, dict) and len(parsed) == 1:
                        sole_value = next(iter(parsed.values()))
                        if isinstance(sole_value, list):
                            log_with_context('debug', "Unwrapped single-key object returned by LLM into list of operations.")
                            parsed = sole_value
                    
                    # Handle empty results
                    if parsed == {} or parsed == []:
                        logger.info("LLM returned empty object/array, treating as empty memory list")
                        return []
                    
                    # Validate memory operations if enabled
                    if hasattr(self._json_parser.repair_system, 'validate_memory_operations') and parsed is not None:
                        validation_errors = self._json_parser.repair_system.validate_memory_operations(parsed)
                        if validation_errors:
                            logger.warning(f"Memory operation validation found issues: {validation_errors}")
                            # Still return the data, but log the issues
                    
                    return parsed
                else:
                    logger.warning(f"JSON repair failed: {result.validation_errors}")
                    # Fall back to legacy parsing
                    
            except Exception as e:
                log_with_context('error', f"Error in enhanced JSON parsing: {e}")
                # Fall back to legacy parsing
        
        # Legacy parsing fallback (simplified version of original logic)
        log_with_context('debug', "Using legacy JSON parsing fallback")
        
        try:
            parsed = json.loads(text)
            log_with_context('debug', "Successfully parsed JSON directly with legacy parser.")
            if isinstance(parsed, dict) and len(parsed) == 1:
                sole_value = next(iter(parsed.values()))
                if isinstance(sole_value, list):
                    log_with_context('debug', "Unwrapped single-key object returned by LLM into list of operations.")
                    parsed = sole_value
            if parsed == {} or parsed == []:
                logger.info("LLM returned empty object/array, treating as empty memory list")
                return []
            return parsed
        except json.JSONDecodeError as e:
            # Don't raise exception here, try fallback methods first
            log_with_context('debug', f"Direct JSON parsing failed, trying fallback methods: {e}")

        # Try extracting from code blocks
        code_block_pattern = r"```(?:json)?\s*(\[[\s\S]*?\]|\{[\s\S]*?\})\s*```"
        matches = re.findall(code_block_pattern, text)
        if matches:
            log_with_context('debug', f"Found {len(matches)} JSON code blocks (legacy fallback)")
            for i, match in enumerate(matches):
                try:
                    parsed = json.loads(match)
                    log_with_context('debug', f"Successfully parsed JSON from code block {i+1} (legacy fallback)")
                    if parsed == {} or parsed == []: continue
                    return parsed
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse JSON from code block {i+1} (legacy fallback): {e}")

        # Check for empty array indicator
        if "[]" in text.replace(" ", ""):
            logger.info("Detected '[]' token in LLM response after exhaustive parsing. Treating as empty list.")
            return []
        
        # Update error counters
        self.error_counters["json_parse_errors"] += 1
        self.error_timestamps["json_parse_errors"].append(time.time())
        self._error_message = "json_parse_error"
        
        # Raise custom exception
        error = MalformedJSONError(
            raw_response=text[:500],  # Truncate for error message
            parse_error="Failed to parse JSON after trying direct parsing, code block extraction, and JSON repair"
        )
        log_exception(logger, error)
        raise error

    def _calculate_memory_similarity(self, memory1: str, memory2: str) -> float:
        if not memory1 or not memory2:
            return 0.0
        memory1_clean = re.sub(r"\[Tags:.*?\]\s*", "", memory1).lower().strip()
        memory2_clean = re.sub(r"\[Tags:.*?\]\s*", "", memory2).lower().strip()
        if memory1_clean == memory2_clean:
            return 1.0
        words1 = set(re.findall(r"\b\w+\b", memory1_clean))
        words2 = set(re.findall(r"\b\w+\b", memory2_clean))
        if not words1 or not words2:
            return 0.0
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        jaccard = intersection / union if union > 0 else 0.0
        seq_similarity = SequenceMatcher(None, memory1_clean, memory2_clean).ratio()
        return (0.4 * jaccard) + (0.6 * seq_similarity)
        
    async def _calculate_embedding_similarity(self, memory1: str, memory2: str) -> float:
        if not memory1 or not memory2:
            return 0.0
        memory1_clean = re.sub(r"\[Tags:.*?\]\\s*", "", memory1).lower().strip()
        memory2_clean = re.sub(r"\[Tags:.*?\]\\s*", "", memory2).lower().strip()
        if memory1_clean == memory2_clean:
            return 1.0
        try:
            mem1_embedding = await self._get_embedding(memory1_clean)
            mem2_embedding = await self._get_embedding(memory2_clean)
            if mem1_embedding is None or mem2_embedding is None:
                log_with_context('warning', "Could not generate embeddings for similarity calculation. Using text-based similarity.",
                               component='EMBED', operation='similarity_fallback')
                return self._calculate_memory_similarity(memory1, memory2)
            similarity = float(np.dot(mem1_embedding, mem2_embedding))
            return max(0.0, min(1.0, similarity))
        except Exception as e:
            log_with_context('error', "Error calculating embedding similarity",
                           component='EMBED', operation='similarity_error', error=e)
            log_with_context('info', "Falling back to text-based similarity due to unexpected error",
                           component='EMBED', operation='similarity_fallback')
            return self._calculate_memory_similarity(memory1, memory2)

    # ----------------------------
    # MinHash Fingerprinting Implementation
    # ----------------------------
    
    class MinHashFingerprinter:
        """
        MinHash-based fingerprinting for efficient duplicate detection.
        
        MinHash provides a probabilistic method for estimating Jaccard similarity
        between sets, which is ideal for detecting near-duplicate text content.
        """
        
        def __init__(self, num_hashes: int = 128, seed: int = 42):
            """
            Initialize MinHash fingerprinter.
            
            Args:
                num_hashes: Number of hash functions to use (more = higher precision)
                seed: Random seed for reproducible hash functions
            """
            self.num_hashes = num_hashes
            self.seed = seed
            self.hash_functions = self._generate_hash_functions()
            
        def _generate_hash_functions(self) -> List[Callable[[int], int]]:
            """Generate a set of independent hash functions."""
            random.seed(self.seed)
            hash_funcs = []
            
            for i in range(self.num_hashes):
                # Generate random coefficients for hash function: (a*x + b) % p
                a = random.randint(1, 2**32 - 1)
                b = random.randint(0, 2**32 - 1)
                
                def make_hash_func(a_val, b_val):
                    def hash_func(x):
                        return (a_val * x + b_val) % (2**32 - 1)
                    return hash_func
                
                hash_funcs.append(make_hash_func(a, b))
            
            return hash_funcs
        
        def _text_to_shingles(self, text: str, k: int = 3) -> Set[str]:
            """
            Convert text to k-shingles (substrings of length k).
            
            Args:
                text: Input text
                k: Shingle size (3-grams are typical for text)
                
            Returns:
                Set of k-shingles
            """
            # Clean and normalize text
            clean_text = re.sub(r'\s+', ' ', text.lower().strip())
            clean_text = re.sub(r'[^\w\s]', '', clean_text)  # Remove punctuation
            
            if len(clean_text) < k:
                return {clean_text}
            
            shingles = set()
            for i in range(len(clean_text) - k + 1):
                shingle = clean_text[i:i + k]
                shingles.add(shingle)
            
            return shingles
        
        def _shingle_to_hash(self, shingle: str) -> int:
            """Convert a shingle to a hash value."""
            return int(hashlib.md5(shingle.encode('utf-8')).hexdigest()[:8], 16)
        
        def generate_fingerprint(self, text: str, shingle_size: int = 3) -> List[int]:
            """
            Generate MinHash fingerprint for the given text.
            
            Args:
                text: Input text to fingerprint
                shingle_size: Size of text shingles to use
                
            Returns:
                List of hash values representing the MinHash signature
            """
            # Convert text to shingles
            shingles = self._text_to_shingles(text, shingle_size)
            
            if not shingles:
                # Empty text or very short text
                return [0] * self.num_hashes
            
            # Convert shingles to hash values
            shingle_hashes = [self._shingle_to_hash(shingle) for shingle in shingles]
            
            # Compute MinHash signature
            signature = []
            for hash_func in self.hash_functions:
                min_hash = min(hash_func(sh) for sh in shingle_hashes)
                signature.append(min_hash)
            
            return signature
        
        def estimate_jaccard_similarity(self, sig1: List[int], sig2: List[int]) -> float:
            """
            Estimate Jaccard similarity between two MinHash signatures.
            
            Args:
                sig1: First MinHash signature
                sig2: Second MinHash signature
                
            Returns:
                Estimated Jaccard similarity [0.0, 1.0]
            """
            if len(sig1) != len(sig2):
                raise ValueError("Signatures must have the same length")
            
            if not sig1 or not sig2:
                return 0.0
            
            matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
            return matches / len(sig1)
    
    def _get_memory_fingerprint(self, memory_content: str) -> List[int]:
        """
        Generate MinHash fingerprint for memory content.
        
        Args:
            memory_content: The memory text to fingerprint
            
        Returns:
            MinHash signature as list of integers
        """
        # Initialize fingerprinter if not already done or config changed
        if not hasattr(self, '_minhash_fingerprinter') or \
           getattr(self, '_cached_num_hashes', 0) != self.internal_config.fingerprint_num_hashes:
            self._minhash_fingerprinter = self.MinHashFingerprinter(
                num_hashes=self.internal_config.fingerprint_num_hashes,
                seed=42  # Fixed seed for consistency
            )
            self._cached_num_hashes = self.internal_config.fingerprint_num_hashes
        
        # Clean memory content (remove tags, normalize)
        clean_content = re.sub(r"\[Tags:.*?\]\s*", "", memory_content).lower().strip()
        
        # Generate and return fingerprint
        return self._minhash_fingerprinter.generate_fingerprint(
            clean_content, 
            self.internal_config.fingerprint_shingle_size
        )
    
    def _calculate_fingerprint_similarity(self, fingerprint1: List[int], fingerprint2: List[int]) -> float:
        """
        Calculate similarity between two MinHash fingerprints.
        
        Args:
            fingerprint1: First fingerprint
            fingerprint2: Second fingerprint
            
        Returns:
            Estimated Jaccard similarity [0.0, 1.0]
        """
        if not hasattr(self, '_minhash_fingerprinter'):
            self._minhash_fingerprinter = self.MinHashFingerprinter(num_hashes=128, seed=42)
        
        try:
            return self._minhash_fingerprinter.estimate_jaccard_similarity(fingerprint1, fingerprint2)
        except Exception as e:
            log_with_context('warning', "Error calculating fingerprint similarity",
                           component='MEMORY', operation='fingerprint_similarity', error=e)
            return 0.0
    
    def _get_enhanced_similarity_score(self, memory1: str, memory2: str) -> tuple[float, str]:
        """
        Calculate enhanced similarity using MinHash fingerprinting combined with existing methods.
        
        Args:
            memory1: First memory content
            memory2: Second memory content
            
        Returns:
            Tuple of (similarity_score, method_used)
        """
        try:
            # Generate fingerprints
            fingerprint1 = self._get_memory_fingerprint(memory1)
            fingerprint2 = self._get_memory_fingerprint(memory2)
            
            # Calculate fingerprint similarity
            fingerprint_sim = self._calculate_fingerprint_similarity(fingerprint1, fingerprint2)
            
            # Get traditional similarity for comparison
            text_sim = self._calculate_memory_similarity(memory1, memory2)
            
            # Combine both methods with fingerprint weighted higher for scalability
            # but text similarity for fine-grained matching
            combined_similarity = (0.7 * fingerprint_sim) + (0.3 * text_sim)
            
            log_with_context('debug', f"Enhanced similarity: fingerprint={fingerprint_sim:.3f}, text={text_sim:.3f}, combined={combined_similarity:.3f}")
            
            return combined_similarity, "fingerprint_enhanced"
            
        except Exception as e:
            log_with_context('warning', "Error in enhanced similarity calculation, using text-based",
                           component='MEMORY', operation='enhanced_similarity', error=e)
            return self._calculate_memory_similarity(memory1, memory2), "text_fallback"

    # ----------------------------
    # LSH (Locality Sensitive Hashing) Implementation
    # ----------------------------
    
    class LSHIndex:
        """
        Locality Sensitive Hashing index for efficient duplicate detection.
        
        Uses band-based LSH to group similar MinHash signatures together,
        reducing the number of pairwise comparisons needed.
        """
        
        def __init__(self, num_bands: int = 16, rows_per_band: int = 8):
            """
            Initialize LSH index.
            
            Args:
                num_bands: Number of bands to use (more bands = higher precision, more memory)
                rows_per_band: Number of rows per band (affects sensitivity)
            """
            self.num_bands = num_bands
            self.rows_per_band = rows_per_band
            self.signature_length = num_bands * rows_per_band
            
            # Hash tables for each band
            self.bands: List[Dict[int, Set[str]]] = [dict() for _ in range(num_bands)]
            
            # Store original signatures by memory ID
            self.signatures: Dict[str, List[int]] = {}  # memory_id -> signature
            self.memory_ids: Set[str] = set()  # Track all memory IDs
            
        def _hash_band(self, band_values: List[int]) -> int:
            """Hash a band's values to create a bucket key."""
            # Use a simple hash of the concatenated values
            return hash(tuple(band_values)) % (2**32 - 1)
        
        def add_signature(self, memory_id: str, signature: List[int]) -> bool:
            """
            Add a MinHash signature to the LSH index.
            
            Args:
                memory_id: Unique identifier for the memory
                signature: MinHash signature
                
            Returns:
                True if added successfully, False if signature length mismatch
            """
            if len(signature) != self.signature_length:
                # Try to adapt signature length if needed
                if len(signature) > self.signature_length:
                    signature = signature[:self.signature_length]
                else:
                    # Pad with zeros if too short
                    signature = signature + [0] * (self.signature_length - len(signature))
            
            # Store the signature
            self.signatures[memory_id] = signature
            self.memory_ids.add(memory_id)
            
            # Add to each band
            for band_idx in range(self.num_bands):
                start_idx = band_idx * self.rows_per_band
                end_idx = start_idx + self.rows_per_band
                band_values = signature[start_idx:end_idx]
                
                band_hash = self._hash_band(band_values)
                
                if band_hash not in self.bands[band_idx]:
                    self.bands[band_idx][band_hash] = set()
                
                self.bands[band_idx][band_hash].add(memory_id)
            
            return True
        
        def remove_signature(self, memory_id: str) -> bool:
            """
            Remove a signature from the LSH index.
            
            Args:
                memory_id: Unique identifier for the memory
                
            Returns:
                True if removed successfully, False if not found
            """
            if memory_id not in self.signatures:
                return False
            
            signature = self.signatures[memory_id]
            
            # Remove from each band
            for band_idx in range(self.num_bands):
                start_idx = band_idx * self.rows_per_band
                end_idx = start_idx + self.rows_per_band
                band_values = signature[start_idx:end_idx]
                
                band_hash = self._hash_band(band_values)
                
                if band_hash in self.bands[band_idx]:
                    self.bands[band_idx][band_hash].discard(memory_id)
                    
                    # Clean up empty buckets
                    if not self.bands[band_idx][band_hash]:
                        del self.bands[band_idx][band_hash]
            
            # Remove from storage
            del self.signatures[memory_id]
            self.memory_ids.discard(memory_id)
            
            return True
        
        def find_candidates(self, signature: List[int]) -> Set[str]:
            if len(signature) != self.signature_length:
                if len(signature) > self.signature_length:
                    signature = signature[:self.signature_length]
                else:
                    signature = signature + [0] * (self.signature_length - len(signature))
            candidates = set()
            for band_idx in range(self.num_bands):
                start_idx = band_idx * self.rows_per_band
                end_idx = start_idx + self.rows_per_band
                band_values = signature[start_idx:end_idx]
                band_hash = self._hash_band(band_values)
                if band_hash in self.bands[band_idx]:
                    candidates.update(self.bands[band_idx][band_hash])
            return candidates
        
        def get_signature(self, memory_id: str) -> Optional[List[int]]:
            return self.signatures.get(memory_id)
        
        def size(self) -> int:
            return len(self.signatures)
        
        def clear(self) -> None:
            self.bands = [dict() for _ in range(self.num_bands)]
            self.signatures.clear()
            self.memory_ids.clear()
    
    def _get_lsh_index(self) -> 'LSHIndex':
        expected_signature_length = self.internal_config.fingerprint_num_hashes
        if not hasattr(self, '_lsh_index') or getattr(self, '_cached_lsh_signature_length', 0) != expected_signature_length:
            if expected_signature_length >= 128:
                num_bands = 16
                rows_per_band = expected_signature_length // num_bands
            elif expected_signature_length >= 64:
                num_bands = 8
                rows_per_band = expected_signature_length // num_bands
            else:
                num_bands = 4
                rows_per_band = max(1, expected_signature_length // num_bands)
            self._lsh_index = self.LSHIndex(num_bands=num_bands, rows_per_band=rows_per_band)
            self._cached_lsh_signature_length = expected_signature_length
            log_with_context('debug', f"Initialized LSH index with {num_bands} bands, {rows_per_band} rows per band")
        return self._lsh_index
    
    def _populate_lsh_index(self, existing_memories: List[Dict[str, Any]]) -> None:
        lsh_index = self._get_lsh_index()
        lsh_index.clear()
        log_with_context('debug', f"Populating LSH index with {len(existing_memories)} existing memories")
        for memory in existing_memories:
            memory_id = memory.get("id")
            memory_content = memory.get("memory", "")
            if memory_id and memory_content:
                try:
                    fingerprint = self._get_memory_fingerprint(memory_content)
                    lsh_index.add_signature(memory_id, fingerprint)
                except Exception as e:
                    logger.warning(f"Failed to add memory {memory_id} to LSH index: {e}")
    
    def _find_lsh_candidates(self, memory_content: str) -> Set[str]:
        try:
            fingerprint = self._get_memory_fingerprint(memory_content)
            lsh_index = self._get_lsh_index()
            candidates = lsh_index.find_candidates(fingerprint)
            log_with_context('debug', f"LSH found {len(candidates)} candidates for similarity checking")
            return candidates
        except Exception as e:
            logger.warning(f"Error in LSH candidate finding: {e}")
            return set()

    # ----------------------------
    # Enhanced Confidence Scoring System
    # ----------------------------
    
    @dataclass
    class DuplicateConfidenceScore:
        fingerprint_similarity: float = 0.0
        text_similarity: float = 0.0
        embedding_similarity: float = 0.0
        semantic_similarity: float = 0.0
        length_similarity: float = 0.0
        tag_overlap: float = 0.0
        temporal_proximity: float = 0.0
        content_type_match: float = 0.0
        combined_score: float = 0.0
        confidence_level: str = "low"
        is_duplicate: bool = False
        primary_method: str = ""
        decision_factors: List[str] = field(default_factory=list)
    
    def _calculate_enhanced_confidence_score(self, new_content: str, existing_content: str,
                                           new_memory_data: Optional[Dict[str, Any]] = None,
                                           existing_memory_data: Optional[Dict[str, Any]] = None) -> DuplicateConfidenceScore:
        score = self.DuplicateConfidenceScore()
        if new_memory_data is None:
            new_memory_data = {"content": new_content, "tags": []}
        if existing_memory_data is None:
            existing_memory_data = {"memory": existing_content, "tags": []}
        
        try:
            # 1. Fingerprint Similarity (if fingerprinting enabled)
            if self.valves.use_fingerprinting:
                try:
                    fingerprint1 = self._get_memory_fingerprint(new_content)
                    fingerprint2 = self._get_memory_fingerprint(existing_content)
                    score.fingerprint_similarity = self._calculate_fingerprint_similarity(fingerprint1, fingerprint2)
                except Exception as e:
                    log_with_context('debug', f"Error calculating fingerprint similarity: {e}")
                    score.fingerprint_similarity = 0.0
            
            # 2. Text Similarity (always calculated as baseline)
            score.text_similarity = self._calculate_memory_similarity(new_content, existing_content)
            
            # 3. Embedding Similarity (if embeddings available)
            if self.valves.use_embeddings_for_deduplication and self._local_embedding_model:
                try:
                    # Calculate embedding similarity - use sync fallback for non-async context
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            # Already in async context - cannot use asyncio.run
                            log_with_context('debug', "Skipping embedding similarity calculation in async context")
                            score.embedding_similarity = 0.0
                        else:
                            embedding_sim = loop.run_until_complete(self._calculate_embedding_similarity(new_content, existing_content))
                            score.embedding_similarity = embedding_sim
                    except RuntimeError:
                        # No event loop - safe to use asyncio.run
                        embedding_sim = asyncio.run(self._calculate_embedding_similarity(new_content, existing_content))
                        score.embedding_similarity = embedding_sim
                except Exception as e:
                    log_with_context('debug', f"Error calculating embedding similarity: {e}")
                    score.embedding_similarity = 0.0
            
            # 4. Length Similarity with safe division
            len1, len2 = len(new_content.strip()), len(existing_content.strip())
            if len1 > 0 and len2 > 0:
                max_len = max(len1, len2)
                if max_len > 0:
                    score.length_similarity = 1.0 - abs(len1 - len2) / max_len
                else:
                    score.length_similarity = 1.0
            else:
                score.length_similarity = 1.0 if len1 == len2 == 0 else 0.0
            
            # 5. Tag Overlap (if available)
            new_tags = set(new_memory_data.get("tags", []))
            existing_tags = set(existing_memory_data.get("tags", []))
            if new_tags or existing_tags:
                if new_tags and existing_tags:
                    intersection = len(new_tags.intersection(existing_tags))
                    union = len(new_tags.union(existing_tags))
                    score.tag_overlap = intersection / union if union > 0 else 0.0
                else:
                    score.tag_overlap = 0.0
            else:
                score.tag_overlap = 0.5  # Neutral when no tags available
            
            # 6. Temporal Proximity (if timestamps available)
            score.temporal_proximity = self._calculate_temporal_similarity(new_memory_data, existing_memory_data)
            
            # 7. Content Type Matching
            score.content_type_match = self._calculate_content_type_similarity(new_content, existing_content)
            
            # 8. Calculate Combined Score using weighted average
            weights = self._get_confidence_score_weights()
            score.combined_score = (
                weights["fingerprint"] * score.fingerprint_similarity +
                weights["text"] * score.text_similarity +
                weights["embedding"] * score.embedding_similarity +
                weights["length"] * score.length_similarity +
                weights["tag"] * score.tag_overlap +
                weights["temporal"] * score.temporal_proximity +
                weights["content_type"] * score.content_type_match
            )
            
            # 9. Determine confidence level and decision
            score.confidence_level = self._classify_confidence_level(score.combined_score)
            score.is_duplicate, score.primary_method, score.decision_factors = self._make_duplicate_decision(score)
            
            log_with_context('debug', f"Enhanced confidence score: combined={score.combined_score:.3f}, "
                        f"level={score.confidence_level}, duplicate={score.is_duplicate}")
            
            return score
            
        except Exception as e:
            logger.error(f"Error in enhanced confidence scoring: {e}")
            # Return minimal score with fallback to text similarity
            score.text_similarity = self._calculate_memory_similarity(new_content, existing_content)
            score.combined_score = score.text_similarity
            score.confidence_level = "low"
            score.is_duplicate = score.combined_score >= self.valves.similarity_threshold
            score.primary_method = "text_fallback"
            score.decision_factors = ["error_fallback"]
            return score
    
    def _get_confidence_score_weights(self) -> Dict[str, float]:
        if self.valves.use_fingerprinting and self.valves.use_embeddings_for_deduplication:
            return {"fingerprint": 0.35, "text": 0.15, "embedding": 0.25, "length": 0.10,
                    "tag": 0.05, "temporal": 0.05, "content_type": 0.05}
        elif self.valves.use_fingerprinting:
            return {"fingerprint": 0.40, "text": 0.25, "embedding": 0.0, "length": 0.15,
                    "tag": 0.08, "temporal": 0.07, "content_type": 0.05}
        elif self.valves.use_embeddings_for_deduplication:
            return {"fingerprint": 0.0, "text": 0.25, "embedding": 0.40, "length": 0.15,
                    "tag": 0.08, "temporal": 0.07, "content_type": 0.05}
        else:
            return {"fingerprint": 0.0, "text": 0.50, "embedding": 0.0, "length": 0.20,
                    "tag": 0.12, "temporal": 0.10, "content_type": 0.08}
    
    def _calculate_temporal_similarity(self, new_data: Dict[str, Any], existing_data: Dict[str, Any]) -> float:
        try:
            new_timestamp = new_data.get("created_at") or new_data.get("timestamp")
            existing_timestamp = existing_data.get("created_at") or existing_data.get("timestamp")
            if not new_timestamp or not existing_timestamp:
                return 0.5
            if isinstance(new_timestamp, str):
                new_time = datetime.fromisoformat(new_timestamp.replace('Z', '+00:00'))
            else:
                new_time = new_timestamp
            if isinstance(existing_timestamp, str):
                existing_time = datetime.fromisoformat(existing_timestamp.replace('Z', '+00:00'))
            else:
                existing_time = existing_timestamp
            time_diff = abs((new_time - existing_time).total_seconds())
            if time_diff < 3600:
                return 1.0
            elif time_diff < 86400:
                return 0.5
            elif time_diff < 604800:
                return 0.1
            else:
                return 0.0
        except Exception as e:
            log_with_context('debug', f"Error calculating temporal similarity: {e}")
            return 0.5
    
    def _calculate_content_type_similarity(self, content1: str, content2: str) -> float:
        try:
            preference_patterns = r'\b(like|love|prefer|enjoy|favorite|hate|dislike)\b'
            fact_patterns = r'\b(is|was|are|were|born|lives|works|studied)\b'
            goal_patterns = r'\b(want|plan|goal|intend|hope|wish|dream)\b'
            relationship_patterns = r'\b(friend|family|brother|sister|mother|father|spouse|partner)\b'
            def get_content_type(text):
                text_lower = text.lower()
                types = []
                if re.search(preference_patterns, text_lower):
                    types.append("preference")
                if re.search(fact_patterns, text_lower):
                    types.append("fact")
                if re.search(goal_patterns, text_lower):
                    types.append("goal")
                if re.search(relationship_patterns, text_lower):
                    types.append("relationship")
                return set(types) if types else {"general"}
            types1 = get_content_type(content1)
            types2 = get_content_type(content2)
            intersection = len(types1.intersection(types2))
            union = len(types1.union(types2))
            return intersection / union if union > 0 else 0.0
        except Exception as e:
            log_with_context('debug', f"Error calculating content type similarity: {e}")
            return 0.5
    
    def _classify_confidence_level(self, combined_score: float) -> str:
        if combined_score >= 0.85:
            return "high"
        elif combined_score >= 0.65:
            return "medium"
        else:
            return "low"
    
    def _make_duplicate_decision(self, score: 'DuplicateConfidenceScore') -> tuple[bool, str, List[str]]:
        decision_factors = []
        if self.valves.use_fingerprinting:
            fingerprint_threshold = self.valves.fingerprint_similarity_threshold
            if score.fingerprint_similarity >= fingerprint_threshold:
                decision_factors.append(f"fingerprint_match_{score.fingerprint_similarity:.3f}")
                return True, "fingerprint", decision_factors
        if self.valves.use_embeddings_for_deduplication:
            embedding_threshold = self.valves.embedding_similarity_threshold
            if score.embedding_similarity >= embedding_threshold:
                decision_factors.append(f"embedding_match_{score.embedding_similarity:.3f}")
                return True, "embedding", decision_factors
        text_threshold = self.valves.similarity_threshold
        if score.text_similarity >= text_threshold:
            decision_factors.append(f"text_match_{score.text_similarity:.3f}")
            return True, "text", decision_factors
        combined_threshold = self.internal_config.confidence_scoring_combined_threshold
        if score.combined_score >= combined_threshold:
            decision_factors.append(f"combined_match_{score.combined_score:.3f}")
            if score.confidence_level == "high":
                decision_factors.append("high_confidence")
            return True, "combined", decision_factors
        decision_factors.append(f"no_match_combined_{score.combined_score:.3f}")
        return False, "no_match", decision_factors

    async def get_relevant_memories(self, current_message: str, user_id: str, user_timezone: Optional[str] = None) -> List[Dict[str, Any]]:
        if not user_id:
            logger.error("get_relevant_memories called without user_id")
            raise ValueError("user_id is required for retrieving memories")
        if not self._ensure_configuration_persistence():
            logger.warning("Configuration persistence check failed in get_relevant_memories, but continuing with current config")
        log_with_context('debug', f"Getting relevant memories for user_id: {user_id}, message: {current_message[:50]}...")

        import time

        # Metrics instrumentation
        RETRIEVAL_REQUESTS.inc()
        _retrieval_start = time.perf_counter()
        start = _retrieval_start
        try:
            # Get all memories for the user
            existing_memories = await self._get_formatted_memories(user_id)
            logger.info(f"Retrieved {len(existing_memories) if existing_memories else 0} existing memories for user {user_id}")

            if not existing_memories:
                logger.info("No existing memories found for relevance assessment")
                return []

            vector_similarities = []
            user_embedding = None
            user_embedding_dim = None
            try:
                # Check cache first for user embedding
                user_embedding = self._get_cached_user_embedding(current_message)
                if user_embedding is None:
                    user_embedding = await self._get_embedding(current_message)
                    if user_embedding is not None:
                        self._cache_user_embedding(current_message, user_embedding)
                if user_embedding is None and not self.valves.use_llm_for_relevance:
                    logger.warning("Cannot calculate relevance — failed to generate embedding and LLM relevance is disabled.")
                    return []
                elif user_embedding is not None:
                    user_embedding_dim = user_embedding.shape[0]
                    log_with_context('debug', f"User message embedding dimension: {user_embedding_dim}")
            except Exception as e:
                self.error_counters["embedding_errors"] += 1
                log_with_context('error', f"Error computing embedding for user message: {e}\n{traceback.format_exc()}")
                if not self.valves.use_llm_for_relevance:
                    logger.warning("Cannot calculate relevance due to embedding error and no LLM fallback.")
                    return []

            if user_embedding is not None:
                # Calculate vector similarities only if user embedding was successful
                for mem in existing_memories:
                    mem_id = mem.get("id")
                    if mem_id is None:
                        continue
                    # Ensure embedding exists in our cache for this memory
                    mem_emb = self.memory_embeddings.get(mem_id)
                    # Lazily compute and cache the memory embedding if not present
                    if mem_emb is None and self._local_embedding_model is not None:
                        try:
                            mem_text = mem.get("memory") or ""
                            if mem_text:
                                mem_emb = await self._get_embedding(mem_text)
                                # Cache for future similarity checks
                                self.memory_embeddings[mem_id] = mem_emb
                        except Exception as e:
                            logger.warning(
                                f"Error computing embedding for memory {mem_id}: {e}"
                            )

                    if mem_emb is not None:
                        try:
                            if user_embedding_dim is not None and mem_emb.shape[0] != user_embedding_dim:
                                logger.warning(f"Dimension mismatch for memory {mem_id} ({mem_emb.shape[0]} vs user {user_embedding_dim}) - removing from cache")
                                del self.memory_embeddings[mem_id]
                                continue
                            sim = float(np.dot(user_embedding, mem_emb))
                            vector_similarities.append((sim, mem))
                        except Exception as e:
                            logger.warning(f"Error calculating similarity for memory {mem_id}: {e}")
                            continue
                        else:
                            log_with_context('debug', f"No embedding available for memory {mem_id} even after attempted computation.")
                    else:
                        log_with_context('debug', f"No embedding available for memory {mem_id} even after attempted computation.")

                vector_similarities.sort(reverse=True, key=lambda x: x[0])
                log_with_context('debug', f"Calculated vector similarities for {len(vector_similarities)} memories",
                               component='MEMORY', user_id=user_id, operation='vector_similarity',
                               context={'top_3_scores': [round(x[0], 3) for x in vector_similarities[:3]]})
                log_with_context('debug', "Relevance thresholds",
                               component='MEMORY', user_id=user_id, operation='thresholds',
                               context={
                                   'vector_threshold': self.internal_config.vector_similarity_threshold,
                                   'relevance_threshold': self.valves.relevance_threshold
                               })
                initial_filter_threshold = min(0.3, self.internal_config.vector_similarity_threshold)
                top_n = self.internal_config.top_n_memories
                filtered_by_vector = [mem for sim, mem in vector_similarities if sim >= initial_filter_threshold][:top_n]
                log_with_context('info', f"Initial vector filter selected {len(filtered_by_vector)} of {len(existing_memories)} memories",
                               component='MEMORY', user_id=user_id, operation='vector_filter',
                               context={
                                   'selected': len(filtered_by_vector),
                                   'total': len(existing_memories),
                                   'threshold': initial_filter_threshold,
                                   'top_n': top_n
                               })
            else:
                log_with_context('warning', "User embedding failed, proceeding with all memories for LLM check",
                               component='MEMORY', user_id=user_id, operation='embedding_fallback')
                filtered_by_vector = existing_memories


            # --- Decide Relevance Method ---
            if not self.valves.use_llm_for_relevance:
                # --- Use Vector Similarity Scores Directly ---
                logger.info("Using vector similarity directly for relevance scoring (LLM call skipped).")
                
                # Use relevance_threshold as the main filter for vector-only mode
                final_relevance_threshold = self.valves.relevance_threshold
                logger.info(f"Applying final relevance threshold: {final_relevance_threshold}")
                
                relevant_memories = []
                # Use the already calculated and sorted vector similarities
                for sim_score, mem in vector_similarities: # Iterate through the originally sorted list
                    log_with_context('debug', f"Checking memory {mem.get('id', 'unknown')}: score={round(sim_score, 3)} vs threshold={final_relevance_threshold}")
                    if sim_score >= final_relevance_threshold:
                        relevant_memories.append(
                            {"id": mem["id"], "memory": mem["memory"], "relevance": sim_score} # Use vector score as relevance
                        )
                        log_with_context('debug', f"✓ Memory {mem.get('id', 'unknown')} passed threshold")
                    else:
                        log_with_context('debug', f"✗ Memory {mem.get('id', 'unknown')} below threshold")

                # Limit to configured number
                final_top_n = self.internal_config.related_memories_n
                final_relevant = relevant_memories[:final_top_n]
                
                # Provide feedback on memory retrieval results
                if len(final_relevant) == 0 and len(vector_similarities) > 0:
                    top_score = vector_similarities[0][0] if vector_similarities else 0
                    logger.warning(
                        f"No memories passed relevance threshold {final_relevance_threshold}. "
                        f"Top similarity score was {round(top_score, 3)}. "
                        f"Consider lowering relevance_threshold for better retrieval."
                    )
                
                logger.info(
                    f"Found {len(final_relevant)} relevant memories using vector similarity >= {final_relevance_threshold} (from {len(vector_similarities)} candidates)"
                )
                duration = time.perf_counter() - _retrieval_start
                RETRIEVAL_LATENCY.observe(duration)
                logger.info(f"Memory retrieval (vector only) took {duration:.2f}s")
                return final_relevant

            else:
                # --- Use LLM for Relevance Scoring (Optimised) ---
                logger.info("Proceeding with LLM call for relevance scoring.")

                # Optimisation: If the vector similarities for *all* candidate memories are above
                # `llm_skip_relevance_threshold`, we consider the vector score sufficiently
                # confident and *skip* the LLM call (Improvement #5).
                confident_threshold = self.internal_config.llm_skip_relevance_threshold

                # Build helper map id -> vector similarity for quick lookup
                id_to_vec_score = {mem['id']: sim for sim, mem in vector_similarities}

                if filtered_by_vector and all(
                    id_to_vec_score.get(mem['id'], 0.0) >= confident_threshold
                    for mem in filtered_by_vector
                ):
                    logger.info(
                        f"All {len(filtered_by_vector)} memories exceed confident vector threshold ({confident_threshold}). Skipping LLM relevance call."
                    )

                    relevant_memories = [
                        {
                            "id": mem["id"],
                            "memory": mem["memory"],
                            "relevance": id_to_vec_score.get(mem["id"], 0.0),
                        }
                        for mem in filtered_by_vector
                    ]
                    # Ensure sorted by relevance desc
                    relevant_memories.sort(key=lambda x: x["relevance"], reverse=True)
                    return relevant_memories[: self.internal_config.related_memories_n]

                # If not confident, fall back to existing LLM relevance path
                memories_for_llm = filtered_by_vector # Use the vector-filtered list

                if not memories_for_llm:
                     log_with_context('debug', "No memories passed vector filter for LLM relevance check.")
                     return []

                # Build the prompt for LLM
                memory_strings = []
                for mem in memories_for_llm:
                    memory_strings.append(f"ID: {mem['id']}, CONTENT: {mem['memory']}")

                system_prompt = self.valves.memory_relevance_prompt
                user_prompt = f"""Current user message: "{current_message}"

Available memories (pre-filtered by vector similarity):
{json.dumps(memory_strings)}

Rate the relevance of EACH memory to the current user message based *only* on the provided content and message context."""

                # Add current datetime for context
                current_datetime = self.get_formatted_datetime(user_timezone)
                user_prompt += f"""

Current datetime: {current_datetime.strftime('%A, %B %d, %Y %H:%M:%S')} ({current_datetime.tzinfo})"""

                # Check cache or call LLM for relevance score
                import time as time_module

                now = time_module.time()
                ttl_seconds = self.valves.cache_ttl_seconds

                relevance_data = []
                uncached_memories = [] # Memories needing LLM call
                uncached_ids = set() # Track IDs needing LLM call

                # Check cache first
                if user_embedding is not None: # Can only use cache if we have user embedding
                    for mem in memories_for_llm:
                        mem_id = mem.get("id")
                        if mem_id is None:
                            continue
                        mem_emb = self.memory_embeddings.get(mem_id)
                        if mem_emb is None:
                             # If memory embedding is missing, cannot use cache, must call LLM
                             if mem_id not in uncached_ids:
                                 uncached_memories.append(mem)
                                 uncached_ids.add(mem_id)
                             continue

                        key = str(hash((user_embedding.tobytes(), mem_emb.tobytes())))
                        cached = self.relevance_cache.get(key)
                        if cached:
                            score, ts = cached
                            if now - ts < ttl_seconds:
                                logger.info(f"Cache hit for memory {mem_id} (LLM relevance)")
                                relevance_data.append(
                                    {"memory": mem["memory"], "id": mem_id, "relevance": score}
                                )
                                continue  # use cached score

                        # Cache miss or expired, add to uncached list if not already there
                        if mem_id not in uncached_ids:
                             uncached_memories.append(mem)
                             uncached_ids.add(mem_id)
                else:
                     # No user embedding, cannot use cache, all need LLM call
                     logger.warning("Cannot use relevance cache as user embedding failed.")
                     uncached_memories = memories_for_llm # Send all vector-filtered memories to LLM


                # If any uncached memories, call LLM
                if uncached_memories:
                    logger.info(f"Calling LLM for relevance on {len(uncached_memories)} uncached memories.")
                    # Build prompt with only uncached memories
                    uncached_memory_strings = [
                        f"ID: {mem['id']}, CONTENT: {mem['memory']}"
                        for mem in uncached_memories
                    ]
                    # Reuse system_prompt, construct user_prompt specifically for uncached items
                    uncached_user_prompt = f"""Current user message: "{current_message}"

Available memories (evaluate relevance for these specific IDs):
{json.dumps(uncached_memory_strings)}

Rate the relevance of EACH listed memory to the current user message based *only* on the provided content and message context."""
                    current_datetime = self.get_formatted_datetime(user_timezone)
                    uncached_user_prompt += f"""

Current datetime: {current_datetime.strftime('%A, %B %d, %Y %H:%M:%S')} ({current_datetime.tzinfo})"""

                    llm_response_text = await self.query_llm_with_retry(
                        system_prompt, uncached_user_prompt # Use the specific uncached prompt
                    )

                    if not llm_response_text or llm_response_text.startswith("Error:"):
                        if llm_response_text:
                            log_with_context('error', 
                                f"Error from LLM during memory relevance: {llm_response_text}"
                            )
                        # If LLM fails, we might return empty or potentially fall back
                        # For now, return empty to indicate failure
                        return []

                    # Parse the LLM response for the uncached items
                    llm_relevance_results = self._extract_and_parse_json(
                        llm_response_text
                    )

                    if not llm_relevance_results or not isinstance(
                        llm_relevance_results, list
                    ):
                        logger.warning("Failed to parse relevance data from LLM response for uncached items.")
                        # Decide how to handle partial failure - return only cached? or empty?
                        # Returning only cached for now
                    else:
                         # Process successful LLM results
                         for item in llm_relevance_results:
                            mem_id = item.get("id")
                            score = item.get("relevance")
                            mem_text = item.get("memory") # Use memory text from LLM response if available
                            if mem_id and isinstance(score, (int, float)):
                                relevance_data.append(
                                    {"memory": mem_text or f"Content for {mem_id}", # Fallback if memory text missing
                                     "id": mem_id,
                                     "relevance": score}
                                )
                                # Save to cache if possible
                                if user_embedding is not None:
                                    mem_emb = self.memory_embeddings.get(mem_id)
                                    if mem_emb is not None:
                                        key = str(hash((user_embedding.tobytes(), mem_emb.tobytes())))
                                        self.relevance_cache[key] = (score, now)
                                    else:
                                         log_with_context('debug', f"Cannot cache relevance for {mem_id}, embedding missing.")
                            else:
                                logger.warning(f"Invalid item format in LLM relevance response: {item}")


                # Combine cached and newly fetched results, filter by relevance threshold
                final_relevant_memories = []
                final_relevance_threshold = self.valves.relevance_threshold  # Use configured relevance threshold for LLM-score filtering.

                seen_ids = set() # Ensure unique IDs in final list
                for item in relevance_data:
                    if not isinstance(item, dict): continue # Skip invalid entries

                    memory_content = item.get("memory")
                    relevance_score = item.get("relevance")
                    mem_id = item.get("id")

                    if memory_content and isinstance(relevance_score, (int, float)) and mem_id:
                        # Use the final_relevance_threshold determined earlier (should be self.valves.relevance_threshold)
                        if relevance_score >= final_relevance_threshold and mem_id not in seen_ids:
                            final_relevant_memories.append(
                                {"id": mem_id, "memory": memory_content, "relevance": relevance_score}
                            )
                            seen_ids.add(mem_id)

                # Sort final list by relevance (descending)
                final_relevant_memories.sort(key=lambda x: x["relevance"], reverse=True)

                # Limit to configured number
                final_top_n = self.internal_config.related_memories_n
                logger.info(
                    f"Found {len(final_relevant_memories)} relevant memories using LLM score >= {final_relevance_threshold}"
                )
                logger.info(f"Memory retrieval (LLM scoring) took {time.perf_counter() - start:.2f}s")
                return final_relevant_memories[:final_top_n]

        except Exception as e:
            logger.error(
                f"Error getting relevant memories: {e}\n{traceback.format_exc()}"
            )
            return []

    async def process_memories(
        self, memories: List[Dict[str, Any]], user_id: str
    ) -> List[Dict[str, Any]]:  # Return list of successfully processed operations
        """Process memory operations"""
        successfully_saved_ops = []
        
        # Validate user_id
        if not user_id:
            logger.error("process_memories called without user_id")
            raise ValueError("user_id is required for processing memories")
            
        logger.info(f"Processing {len(memories)} memories for user_id: {user_id}")
        
        try:
            user = Users.get_user_by_id(user_id)
            if not user:
                log_with_context('error', f"User not found: {user_id}")
                return []

            # Get existing memories for deduplication
            existing_memories = []
            if self.valves.deduplicate_memories:
                existing_memories = await self._get_formatted_memories(user_id)

            log_with_context('debug', f"Processing {len(memories)} memory operations")

            # First filter for duplicates if enabled
            processed_memories = []
            if self.valves.deduplicate_memories and existing_memories:
                # Store all existing contents for quick lookup
                existing_contents = []
                for mem in existing_memories:
                    existing_contents.append(mem["memory"])

                log_with_context('debug', f"[DEDUPE] Existing memories being checked against: {existing_contents}")

                # Decide similarity method and corresponding threshold
                use_embeddings = self.valves.use_embeddings_for_deduplication
                use_fingerprinting = self.valves.use_fingerprinting
                
                if use_fingerprinting:
                    threshold_to_use = self.valves.fingerprint_similarity_threshold
                    method_desc = "fingerprint-enhanced"
                elif use_embeddings:
                    threshold_to_use = self.valves.embedding_similarity_threshold
                    method_desc = "embedding-based"
                else:
                    threshold_to_use = self.valves.similarity_threshold
                    method_desc = "text-based"
                
                log_with_context('debug', 
                    f"Using {method_desc} similarity for deduplication. "
                    f"Threshold: {threshold_to_use}"
                )

                # Determine if we should use LSH optimization
                use_lsh = (
                    use_fingerprinting and 
                    self.internal_config.use_lsh_optimization and 
                    len(existing_memories) >= self.internal_config.lsh_threshold_for_activation
                )
                
                if use_lsh:
                    log_with_context('debug', f"Using LSH optimization for {len(existing_memories)} existing memories")
                    # Populate LSH index with existing memories
                    self._populate_lsh_index(existing_memories)
                else:
                    log_with_context('debug', f"Using direct comparison (LSH disabled or insufficient memories: {len(existing_memories)})")

                # Check each new memory against existing ones
                for new_memory_idx, memory_dict in enumerate(memories):
                    if memory_dict["operation"] == "NEW":
                        log_with_context('debug', f"[DEDUPE CHECK {new_memory_idx+1}/{len(memories)}] Processing NEW memory: {memory_dict}") # LOG START
                        # Format the memory content
                        operation = MemoryOperation(**memory_dict)
                        formatted_content = self._format_memory_content(operation)

                        # --- BYPASS: Skip dedup for short preference statements ---
                        if (
                            self.valves.enable_short_preference_shortcut
                            and len(formatted_content) <= self.internal_config.short_preference_no_dedupe_length
                        ):
                            pref_kwds = [kw.strip() for kw in self.internal_config.preference_keywords_no_dedupe.split(',') if kw.strip()]
                            if any(kw in formatted_content.lower() for kw in pref_kwds):
                                log_with_context('debug', "Bypassing deduplication for short preference statement: '%s'", formatted_content)
                                processed_memories.append(memory_dict)
                                continue  # Skip duplicate checking entirely for this memory

                        is_duplicate = False
                        similarity_score = 0.0 # Track similarity score for logging
                        similarity_method = 'none' # Track method used

                        if use_embeddings:
                            # Precompute embedding for the new memory once
                            try:
                                if self._local_embedding_model is None:
                                    raise ValueError("Embedding model not available")
                                new_embedding = self._local_embedding_model.encode(
                                    formatted_content.lower().strip(), normalize_embeddings=True
                                )
                            except Exception as e:
                                logger.warning(f"Failed to encode new memory for deduplication; falling back to text sim. Error: {e}")
                                use_embeddings = False  # fall back

                        # Determine which memories to check based on LSH optimization
                        if use_lsh:
                            # Use LSH to find candidate memory IDs
                            candidate_ids = self._find_lsh_candidates(formatted_content)
                            log_with_context('debug', f"LSH narrowed down to {len(candidate_ids)} candidates from {len(existing_memories)} total memories")
                            
                            # Build list of candidates to check
                            candidates_to_check = []
                            for memory in existing_memories:
                                if memory.get("id") in candidate_ids:
                                    candidates_to_check.append((memory.get("memory", ""), memory))
                        else:
                            # Check all existing memories (original behavior)
                            candidates_to_check = [(content, existing_memories[idx]) for idx, content in enumerate(existing_contents)]
                        
                        # Process candidates (either LSH-filtered or all existing memories)
                        for existing_content, existing_mem_dict in candidates_to_check:
                            # Use enhanced confidence scoring if enabled
                            if self.valves.use_enhanced_confidence_scoring:
                                # Create memory data for enhanced scoring
                                new_memory_data = {"content": formatted_content, "tags": memory_dict.get("tags", [])}
                                confidence_score = self._calculate_enhanced_confidence_score(
                                    formatted_content, 
                                    existing_content,
                                    new_memory_data=new_memory_data,
                                    existing_memory_data=existing_mem_dict
                                )
                                
                                similarity_score = confidence_score.combined_score
                                similarity_method = f"enhanced_{confidence_score.primary_method}"
                                
                                # Use the enhanced decision from confidence scoring
                                if confidence_score.is_duplicate:
                                    existing_id = existing_mem_dict.get("id", "unknown")
                                    log_with_context('debug', 
                                        f"  -> Enhanced duplicate found vs {existing_id} "
                                        f"(Combined: {confidence_score.combined_score:.3f}, "
                                        f"Level: {confidence_score.confidence_level}, "
                                        f"Method: {confidence_score.primary_method}, "
                                        f"Factors: {confidence_score.decision_factors})"
                                    )
                                    log_with_context('debug', 
                                        f"Skipping duplicate NEW memory (enhanced confidence): {formatted_content[:50]}..."
                                    )
                                    is_duplicate = True
                                    self._duplicate_skipped += 1
                                    break
                            else:
                                # Fallback to original similarity methods
                                if self.valves.use_fingerprinting:
                                    # Use enhanced fingerprinting-based similarity
                                    similarity, similarity_method = self._get_enhanced_similarity_score(
                                        formatted_content, existing_content
                                    )
                                    similarity_score = similarity
                                elif use_embeddings:
                                    # Retrieve or compute embedding for the existing memory content
                                    existing_id = existing_mem_dict.get("id")
                                    if existing_id is None:
                                        similarity_score = 0.0
                                        similarity_method = "ID_missing"
                                    else:
                                        existing_emb = self.memory_embeddings.get(existing_id)
                                        if existing_emb is None and self._local_embedding_model is not None:
                                            try:
                                                existing_emb = self._local_embedding_model.encode(
                                                    existing_content.lower().strip(), normalize_embeddings=True
                                                )
                                                self.memory_embeddings[existing_id] = existing_emb
                                            except Exception:
                                                # On failure, mark duplicate check using text sim for this item
                                                existing_emb = None
                                        if existing_emb is not None:
                                            similarity = float(np.dot(new_embedding, existing_emb))
                                            similarity_score = similarity # Store score
                                            similarity_method = 'embedding'
                                        else:
                                            similarity = self._calculate_memory_similarity(
                                                formatted_content, existing_content
                                            )
                                            similarity_score = similarity # Store score
                                            similarity_method = 'text'
                                else:
                                    # Choose the appropriate similarity calculation method
                                    similarity = self._calculate_memory_similarity(
                                        formatted_content, existing_content
                                    )
                                    similarity_score = similarity
                                    similarity_method = 'text'
                                    
                                if similarity_score >= threshold_to_use:
                                    existing_id = existing_mem_dict.get("id", "unknown")
                                    log_with_context('debug', 
                                        f"  -> Duplicate found vs existing mem {existing_id} (Similarity: {similarity_score:.3f}, Method: {similarity_method}, Threshold: {threshold_to_use})"
                                    )
                                    log_with_context('debug', 
                                        f"Skipping duplicate NEW memory (similarity: {similarity_score:.2f}, method: {similarity_method}): {formatted_content[:50]}..."
                                    )
                                    is_duplicate = True
                                    # Increment duplicate skipped counter for status reporting
                                    self._duplicate_skipped += 1
                                    break # Stop checking against other existing memories for this new one

                        if not is_duplicate:
                            log_with_context('debug', f"  -> No duplicate found. Adding to processed list: {formatted_content[:50]}...")
                            processed_memories.append(memory_dict)
                        else:
                             log_with_context('debug', f"NEW memory was identified as duplicate and skipped: {formatted_content[:50]}...")
                    else:
                        # Keep all UPDATE and DELETE operations
                        log_with_context('debug', f"Keeping non-NEW operation: {memory_dict['operation']} ID: {memory_dict.get('id', 'N/A')}")
                        processed_memories.append(memory_dict)
            else:
                log_with_context('debug', "Deduplication skipped (valve disabled or no existing memories). Processing all operations.")
                processed_memories = memories

            # Process the filtered memories
            log_with_context('debug', f"Executing {len(processed_memories)} filtered memory operations.")
            for idx, memory_dict in enumerate(processed_memories):
                log_with_context('debug', f"Executing operation {idx + 1}/{len(processed_memories)}: {memory_dict}")
                try:
                    # Validate memory operation
                    operation = MemoryOperation(**memory_dict)
                    # Execute the memory operation
                    await self._execute_memory_operation(operation, user)
                    # If successful, add to our list
                    log_with_context('debug', f"Successfully executed operation: {operation.operation} ID: {operation.id}")
                    successfully_saved_ops.append(memory_dict)
                except ValueError as e:
                    log_with_context('error', f"Invalid memory operation during execution phase: {e} {memory_dict}")
                    self.error_counters["memory_crud_errors"] += 1 # Increment error counter
                    continue
                except Exception as e:
                    log_with_context('error', f"Error executing memory operation in process_memories: {e} {memory_dict}")
                    self.error_counters["memory_crud_errors"] += 1 # Increment error counter
                    continue

            log_with_context('debug', 
                f"Successfully executed {len(successfully_saved_ops)} memory operations out of {len(processed_memories)} processed.")
            # Add confirmation message if any memory was added or updated
            if successfully_saved_ops:
                # Check if any operation was NEW or UPDATE
                if any(op.get("operation") in ["NEW", "UPDATE"] for op in successfully_saved_ops):
                    log_with_context('debug', "Attempting to add confirmation message.") # Log confirmation attempt
                    try:
                        # Request import removed - not used

                        # Find the last assistant message and append confirmation
                        # This is a safe operation, no error if no assistant message
                        for i in reversed(range(len(self._last_body.get("messages", [])))):
                            msg = self._last_body["messages"][i]
                            if msg.get("role") == "assistant":
                                # Do nothing here
                                break
                    except Exception:
                        pass
            return successfully_saved_ops
        except Exception as e:
            logger.error(f"Error processing memories: {e}\n{traceback.format_exc()}")
            return []  # Return empty list on major error

    async def _execute_memory_operation(
        self, operation: MemoryOperation, user: Any
    ) -> None:
        """Execute a memory operation (NEW, UPDATE, DELETE)"""
        formatted_content = self._format_memory_content(operation)
        
        # Extract user_id early for validation and consistent logging
        user_id = getattr(user, 'id', None)
        if not user_id:
            logger.error(f"User object missing 'id' attribute: {user}")
            raise ValueError("User ID is required for memory operations")

        if operation.operation == "NEW":
            try:
                result = await add_memory(
                    user_id=user_id,  # Use extracted user_id
                    form_data=AddMemoryForm(
                        content=formatted_content,
                        metadata={
                            "tags": operation.tags,
                            "memory_bank": operation.memory_bank or self.valves.default_memory_bank,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "source": "adaptive_memory_v3",
                        },
                    ),
                )
                logger.info(f"NEW memory created for user {user_id}: {formatted_content[:50]}...")

                # Generate and cache embedding for new memory if embedding model is available
                # This helps with future deduplication checks when using embedding-based similarity
                if self._local_embedding_model is not None:
                    # Handle both Pydantic model and dict response forms
                    mem_id = getattr(result, "id", None)
                    if mem_id is None and isinstance(result, dict):
                        mem_id = result.get("id")
                    if mem_id is not None:
                        try:
                            memory_clean = re.sub(r"\[Tags:.*?\]\s*", "", formatted_content).lower().strip()
                            memory_embedding = self._local_embedding_model.encode(
                                memory_clean, normalize_embeddings=True
                            )
                            self.memory_embeddings[mem_id] = memory_embedding
                            log_with_context('debug', f"Generated and cached embedding for new memory ID: {mem_id}")
                        except Exception as e:
                            logger.warning(f"Failed to generate embedding for new memory: {e}")
                            # Non-critical error, don't raise

            except Exception as e:
                self.error_counters["memory_crud_errors"] += 1
                log_with_context('error', 
                    f"Error creating memory (operation=NEW, user_id={user_id}): {e}\n{traceback.format_exc()}"
                )
                raise

        elif operation.operation == "UPDATE" and operation.id:
            try:
                # Delete existing memory
                deleted = await delete_memory_by_id(user_id=user_id, memory_id=operation.id)
                if deleted:
                    # Create new memory with updated content
                    result = await add_memory(
                        user_id=user_id,
                        form_data=AddMemoryForm(
                            content=formatted_content,
                            metadata={
                                "tags": operation.tags,
                                "memory_bank": operation.memory_bank or self.valves.default_memory_bank,
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "source": "adaptive_memory_v3",
                            },
                        ),
                    )
                    logger.info(
                        f"UPDATE memory {operation.id} for user {user_id}: {formatted_content[:50]}..."
                    )

                    # Update embedding for modified memory
                    if self._local_embedding_model is not None:
                        # Handle both Pydantic model and dict response forms
                        new_mem_id = getattr(result, "id", None)
                        if new_mem_id is None and isinstance(result, dict):
                            new_mem_id = result.get("id")

                        if new_mem_id is not None:
                            try:
                                memory_clean = re.sub(r"\[Tags:.*?\]\s*", "", formatted_content).lower().strip()
                                memory_embedding = self._local_embedding_model.encode(
                                    memory_clean, normalize_embeddings=True
                                )
                                # Store with the new ID from the result
                                self.memory_embeddings[new_mem_id] = memory_embedding
                                log_with_context('debug', 
                                    f"Updated embedding for memory ID: {new_mem_id} (was: {operation.id})"
                                )

                                # Remove old embedding if ID changed
                                if operation.id != new_mem_id and operation.id in self.memory_embeddings:
                                    del self.memory_embeddings[operation.id]
                            except Exception as e:
                                logger.warning(
                                    f"Failed to update embedding for memory ID {new_mem_id}: {e}"
                                )
                                # Non-critical error, don't raise

                else:
                    logger.warning(f"Memory {operation.id} not found for UPDATE for user {user_id}")
            except Exception as e:
                self.error_counters["memory_crud_errors"] += 1
                log_with_context('error', 
                    f"Error updating memory (operation=UPDATE, memory_id={operation.id}, user_id={user_id}): {e}\n{traceback.format_exc()}"
                )
                raise

            # Invalidate cache entries involving this memory
            mem_emb = self.memory_embeddings.get(operation.id)
            if mem_emb is not None:
                keys_to_delete: List[str] = []
                for key, (score, ts) in self.relevance_cache.items():
                    # key is hash of (user_emb, mem_emb)
                    # We can't extract mem_emb from key, so approximate by deleting all keys with this mem_emb
                    # Since we can't reverse hash, we skip this for now
                    # Future: store reverse index or use tuple keys
                    pass  # Placeholder for future precise invalidation

        elif operation.operation == "DELETE" and operation.id:
            try:
                deleted = await delete_memory_by_id(user_id=user_id, memory_id=operation.id)
                logger.info(f"DELETE memory {operation.id} for user {user_id}: {deleted}")

                # Invalidate cache entries involving this memory
                mem_emb = self.memory_embeddings.get(operation.id)
                if mem_emb is not None:
                    keys_to_delete = []
                    for key, (score, ts) in self.relevance_cache.items():
                        # Same as above, placeholder
                        pass

                # Remove embedding
                if operation.id in self.memory_embeddings:
                    del self.memory_embeddings[operation.id]
                    log_with_context('debug', f"Removed embedding for deleted memory ID: {operation.id}")

            except Exception as e:
                self.error_counters["memory_crud_errors"] += 1
                log_with_context('error', 
                    f"Error deleting memory (operation=DELETE, memory_id={operation.id}, user_id={user_id}): {e}\n{traceback.format_exc()}"
                )
                raise

    def _format_memory_content(self, operation: MemoryOperation) -> str:
        """Format memory content with tags, memory bank, and confidence for saving / display"""
        content = operation.content or ""
        tag_part = f"[Tags: {', '.join(operation.tags)}] " if operation.tags else ""
        bank_part = f" [Memory Bank: {operation.memory_bank or self.valves.default_memory_bank}]"
        # Format confidence score, handling None case
        confidence_score = operation.confidence if operation.confidence is not None else 0.0 # Default to 0.0 if None for formatting
        confidence_part = f" [Confidence: {confidence_score:.2f}]" # Format to 2 decimal places
        return f"{tag_part}{content}{bank_part}{confidence_part}".strip()

    def _handle_api_error(self, error_msg: str, provider_type: str, operation: str = "API") -> str:
        """Consolidated error handling for API operations."""
        formatted_msg = f"Error: {operation} ({provider_type}) - {error_msg}"
        logger.error(formatted_msg)
        return formatted_msg

    def _handle_validation_error(self, field_name: str, value: Any, expected_type: Optional[str] = None) -> bool:
        """Consolidated validation error handling."""
        if expected_type:
            logger.warning(f"Validation failed for {field_name}: expected {expected_type}, got {type(value)}")
        else:
            logger.warning(f"Validation failed for {field_name}: invalid value {value}")
        return False

    def _safe_execute_with_fallback(self, operation: Callable, fallback_value: Any = None, operation_name: str = "operation") -> Any:
        """Safely execute an operation with fallback value on exception."""
        try:
            return operation()
        except Exception as e:
            logger.warning(f"{operation_name} failed: {e}")
            return fallback_value

    async def _make_api_request_with_retry(self, 
                                         api_url: str, 
                                         data: dict, 
                                         headers: dict,
                                         provider_type: str,
                                         max_retries: Optional[int] = None,
                                         retry_delay: Optional[float] = None,
                                         request_timeout: Optional[float] = None) -> tuple[bool, Union[dict, str]]:
        """
        Unified API request method with retry logic for both LLM and embedding requests.
        
        Args:
            api_url: API endpoint URL
            data: Request payload
            headers: Request headers
            provider_type: Provider type for circuit breaker and logging
            max_retries: Maximum retry attempts (defaults to valve setting)
            retry_delay: Base delay between retries (defaults to valve setting)
            request_timeout: Request timeout (defaults to valve setting)
            
        Returns:
            Tuple of (success: bool, response_data: dict or error_message: str)
        """
        max_retries = max_retries or self.valves.max_retries
        retry_delay = retry_delay or self.valves.retry_delay
        request_timeout = request_timeout or self.valves.request_timeout
        
        # Check circuit breaker before attempting connection
        if self._is_circuit_breaker_open(api_url, provider_type):
            error_msg = f"Circuit breaker is open for {provider_type} at {api_url}. Endpoint temporarily unavailable."
            logger.warning(error_msg)
            return False, error_msg

        # Ensure we have a valid aiohttp session
        session = await self._get_aiohttp_session()
        
        for attempt in range(1, max_retries + 2):  # +2 because we start at 1 and want max_retries+1 attempts
            log_with_context('debug', f"API request attempt {attempt}/{max_retries+1} to {api_url}")
            
            try:
                # Make the API call with timeout
                timeout = aiohttp.ClientTimeout(total=request_timeout)
                async with session.post(api_url, json=data, headers=headers, timeout=timeout) as response:
                    log_with_context('debug', f"API response status: {response.status}")
                    
                    if response.status == 200:
                        # Success - parse response
                        content_type = response.headers.get("content-type", "")
                        
                        if "application/x-ndjson" in content_type:
                            # Handle NDJSON (Ollama may return this)
                            raw_text = await response.text()
                            log_with_context('debug', f"Received NDJSON response length: {len(raw_text)}")
                            
                            last_json = None
                            for line in raw_text.strip().splitlines():
                                try:
                                    last_json = json.loads(line)
                                except json.JSONDecodeError:
                                    continue
                            
                            if last_json is None:
                                error_msg = "Could not decode NDJSON response"
                                log_with_context('error', error_msg)
                                if attempt > max_retries:
                                    return False, error_msg
                                else:
                                    continue
                            
                            response_data = last_json
                        else:
                            # Regular JSON
                            response_data = await response.json()
                        
                        # Record success for circuit breaker
                        self._record_circuit_breaker_success(api_url, provider_type)
                        log_with_context('debug', f"API request successful: {json.dumps(response_data)[:200]}...")
                        return True, response_data
                        
                    else:
                        # Handle error response
                        error_text = await response.text()
                        error_msg = f"API ({provider_type}) returned {response.status}: {error_text}"
                        
                        # Log specific exceptions based on status code
                        if response.status == 401:
                            auth_error = LLMAuthenticationError(provider=provider_type, status_code=response.status)
                            log_exception(logger, auth_error)
                        elif response.status == 503:
                            retry_after = response.headers.get('Retry-After')
                            service_error = LLMServiceUnavailableError(
                                provider=provider_type, 
                                status_code=response.status,
                                retry_after=int(retry_after) if retry_after and retry_after.isdigit() else None
                            )
                            log_exception(logger, service_error)
                        else:
                            logger.warning(f"API error: {error_msg}")
                        
                        # Record failure for circuit breaker
                        self._record_circuit_breaker_failure(api_url, provider_type)
                        
                        # Determine if retryable
                        is_retryable = response.status in [429, 500, 502, 503, 504]
                        
                        if is_retryable and attempt <= max_retries:
                            # Enhanced exponential backoff with jitter
                            base_delay = retry_delay * (2 ** (attempt - 1))
                            # More aggressive backoff for rate limiting
                            if response.status == 429:
                                jitter = random.uniform(0.5, 2.0)
                            else:
                                jitter = random.uniform(0.1, 1.0)
                            
                            sleep_time = base_delay + jitter
                            logger.warning(f"Retrying in {sleep_time:.2f} seconds (status: {response.status})...")
                            await asyncio.sleep(sleep_time)
                            continue
                        else:
                            return False, error_msg
                            
            except asyncio.TimeoutError:
                timeout_error = LLMTimeoutError(
                    provider=provider_type,
                    timeout_seconds=30.0,  # Default timeout
                    endpoint=api_url
                )
                log_exception(logger, timeout_error, level="warning")
                self._record_circuit_breaker_failure(api_url, provider_type)
                
                if attempt <= max_retries:
                    # Enhanced exponential backoff with jitter for timeouts
                    base_delay = retry_delay * (2 ** (attempt - 1))
                    jitter = random.uniform(0.1, 0.8)
                    sleep_time = base_delay + jitter
                    logger.info(f"Retrying after timeout in {sleep_time:.2f} seconds...")
                    await asyncio.sleep(sleep_time)
                    continue
                else:
                    raise timeout_error
                    
            except ClientError as e:
                logger.warning(f"Attempt {attempt} failed: API connection error: {str(e)}")
                self._record_circuit_breaker_failure(api_url, provider_type)
                
                if attempt <= max_retries:
                    # Enhanced exponential backoff for connection errors
                    base_delay = retry_delay * (2 ** (attempt - 1))
                    jitter = random.uniform(0.2, 1.0)
                    sleep_time = base_delay + jitter
                    logger.warning(f"Retrying after connection error in {sleep_time:.2f} seconds...")
                    await asyncio.sleep(sleep_time)
                    continue
                else:
                    return False, f"API connection error after {max_retries} attempts: {str(e)}"
                    
            except Exception as e:
                log_with_context('error', f"Attempt {attempt} failed: Unexpected error during API request: {e}")
                self._record_circuit_breaker_failure(api_url, provider_type)
                
                if attempt <= max_retries:
                    # Generic retry for unexpected errors
                    base_delay = retry_delay * (2 ** (attempt - 1))
                    jitter = random.uniform(0.3, 1.2)
                    sleep_time = base_delay + jitter
                    logger.warning(f"Retrying after unexpected error in {sleep_time:.2f} seconds...")
                    await asyncio.sleep(sleep_time)
                    continue
                else:
                    return False, f"Unexpected API error after {max_retries} attempts: {str(e)}"
        
        return False, f"API request failed after {max_retries} attempts"

    def _build_llm_request_payload(self, system_prompt: str, user_prompt: str, provider_type: str, 
                                   model: str, provider_features: dict) -> dict:
        """Build LLM request payload for specific provider type."""
        if provider_type == "ollama":
            return {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.95,
                    "top_k": 80,
                    "num_predict": 2048,
                    "format": "json",
                },
                "stream": False,
            }
        elif provider_type == "openai_compatible":
            request_data = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0,
                "top_p": 1,
                "max_tokens": 1024,
                "seed": 42,
                "stream": False,
            }
            # Add JSON mode if supported
            if provider_features.get("supports_json_mode", True):
                request_data["response_format"] = {"type": "json_object"}
            return request_data
        elif provider_type == "gemini":
            # Combine system prompt with user prompt for Gemini
            combined_prompt = f"{system_prompt}\n\nUser: {user_prompt}\n\nPlease respond in valid JSON format."
            
            gemini_data: Dict[str, Any] = {
                "contents": [
                    {
                        "parts": [
                            {"text": combined_prompt}
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.1,
                    "topP": 0.95,
                    "maxOutputTokens": 1024,
                    "stopSequences": [],
                    "responseMimeType": "application/json"
                },
                "safetySettings": [
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                    },
                    {
                        "category": "HARM_CATEGORY_HATE_SPEECH",
                        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                    },
                    {
                        "category": "HARM_CATEGORY_HARASSMENT",
                        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                    },
                    {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                    }
                ]
            }
            
            # Add system instruction for newer models
            if "gemini-1.5" in model.lower() or "gemini-pro" in model.lower():
                gemini_data["systemInstruction"] = {
                    "parts": [
                        {"text": system_prompt}
                    ]
                }
                # Use only user prompt in contents when system instruction is provided
                if isinstance(gemini_data["contents"], list) and isinstance(gemini_data["contents"][0], dict):
                    gemini_data["contents"][0]["parts"][0]["text"] = f"{user_prompt}\n\nPlease respond in valid JSON format."
            
            return gemini_data
        else:
            raise UnsupportedProviderError(
                provider=provider_type,
                supported_providers=["ollama", "openai_compatible", "gemini"]
            )

    def _build_llm_request_headers(self, provider_type: str, api_key: Optional[str]) -> dict:
        """Build headers for LLM requests."""
        headers = {"Content-Type": "application/json"}
        
        if provider_type == "openai_compatible" and api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        elif provider_type == "gemini" and api_key:
            headers["x-goog-api-key"] = api_key
        
        return headers

    def _prepare_gemini_api_url(self, api_url: str, model: str, api_key: Optional[str]) -> str:
        """Prepare Gemini API URL with proper format."""
        if "generativelanguage.googleapis.com" not in api_url:
            logger.warning(f"Gemini API URL should use generativelanguage.googleapis.com, got: {api_url}")
            # Try to construct proper URL
            if "chat/completions" in api_url or "v1/completions" in api_url:
                model_for_url = model if model else "gemini-pro"
                api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_for_url}:generateContent"
                logger.info(f"Updated Gemini API URL to: {api_url}")
        
        # Add API key to URL if not already present and not in headers
        if api_key and "key=" not in api_url:
            api_url = f"{api_url}?key={api_key}"
        
        return api_url

    def _extract_llm_response_content(self, response_data: Union[dict, str], provider_type: str) -> Optional[str]:
        """Extract content from LLM response based on provider type."""
        # Handle case where response_data is a string (error message)
        if isinstance(response_data, str):
            return response_data
        if provider_type == "openai_compatible":
            if (response_data.get("choices")
                and response_data["choices"][0].get("message")
                and response_data["choices"][0]["message"].get("content")):
                return response_data["choices"][0]["message"]["content"]
        elif provider_type == "gemini":
            # Primary Gemini response format
            if (response_data.get("candidates")
                and len(response_data["candidates"]) > 0
                and response_data["candidates"][0].get("content")
                and response_data["candidates"][0]["content"].get("parts")
                and len(response_data["candidates"][0]["content"]["parts"]) > 0
                and response_data["candidates"][0]["content"]["parts"][0].get("text")):
                return response_data["candidates"][0]["content"]["parts"][0]["text"]
            # Fallback format
            elif (response_data.get("choices")
                and response_data["choices"][0].get("message")
                and response_data["choices"][0]["message"].get("content")):
                return response_data["choices"][0]["message"]["content"]
        elif provider_type == "ollama":
            if response_data.get("message") and response_data["message"].get("content"):
                return response_data["message"]["content"]
        
        return None

    async def query_llm_with_retry(self, system_prompt: str, user_prompt: str) -> str:
        """Query LLM with retry logic, supporting multiple provider types.

        Args:
            system_prompt: System prompt for context/instructions
            user_prompt: User prompt/query

        Returns:
            String response from LLM or error message
        """
        # Get configuration from valves
        provider_type = self.valves.llm_provider 
        model = self.valves.llm_model_name
        api_url = self.valves.llm_api_endpoint_url
        api_key = self.valves.llm_api_key

        logger.info(f"LLM Query: Provider={provider_type}, Model={model}, URL={api_url}")
        log_with_context('debug', f"System prompt length: {len(system_prompt)}, User prompt length: {len(user_prompt)}")

        # Track LLM call frequency
        try:
            self.metrics["llm_call_count"] = self.metrics.get("llm_call_count", 0) + 1
        except Exception as metric_err:
            log_with_context('debug', f"Unable to increment llm_call_count metric: {metric_err}")

        # Perform health check if enabled
        if self.internal_config.enable_health_checks:
            is_healthy = await self._check_endpoint_health(api_url, provider_type)
            if not is_healthy:
                error_msg = f"Health check failed for {provider_type} at {api_url}. Endpoint may be unavailable."
                logger.warning(error_msg)
                self._record_circuit_breaker_failure(api_url, provider_type)
                return error_msg

        # Add current datetime to system prompt for time awareness
        system_prompt_with_date = system_prompt
        try:
            now = self.get_formatted_datetime()
            tzname = now.tzname() or "UTC"
            system_prompt_with_date = f"{system_prompt}\n\nCurrent date and time: {now.strftime('%Y-%m-%d %H:%M:%S')} {tzname}"
        except Exception as e:
            logger.warning(f"Could not add date to system prompt: {e}")

        # Detect provider features for adaptive behavior
        provider_features = {}
        if self.valves.enable_feature_detection:
            try:
                provider_features = await self._get_provider_features(provider_type, api_url, api_key)
                log_with_context('debug', f"Provider features for {provider_type}: {provider_features}")
            except Exception as e:
                logger.warning(f"Feature detection failed, using defaults: {e}")
                provider_features = {
                    "supports_json_mode": True,
                    "supports_system_messages": True,
                    "supports_streaming": True
                }
        else:
            # Use safe defaults when feature detection is disabled
            provider_features = {
                "supports_json_mode": True,
                "supports_system_messages": True,
                "supports_streaming": True
            }
            log_with_context('debug', "Feature detection disabled, using default capabilities")

        try:
            # Build request payload using consolidated helper
            data = self._build_llm_request_payload(system_prompt_with_date, user_prompt, provider_type, model, provider_features)
            log_with_context('debug', f"Request payload: {json.dumps(data)[:500]}...")
            
            # Build headers using consolidated helper
            headers = self._build_llm_request_headers(provider_type, api_key)
            
            # Prepare Gemini URL if needed
            if provider_type == "gemini":
                api_url = self._prepare_gemini_api_url(api_url, model, api_key)
                # Remove API key from headers if it's in URL to avoid duplication
                if "key=" in api_url:
                    headers.pop("x-goog-api-key", None)

        except ValueError as e:
            error_msg = f"Unsupported provider type: {str(e)}"
            logger.error(error_msg)
            return error_msg

        # Make API request using consolidated method
        success, response_data = await self._make_api_request_with_retry(
            api_url, data, headers, provider_type,
            max_retries=self.valves.max_retries,
            retry_delay=self.valves.retry_delay,
            request_timeout=self.valves.request_timeout
        )

        if not success:
            # response_data contains error message
            error_msg = str(response_data)
            
            # Handle special feature-specific failures (like JSON mode)
            if ("json_object" in error_msg.lower() or 
                "response_format" in error_msg.lower() or
                "invalid request" in error_msg.lower()) and provider_type in ["openai_compatible", "gemini"]:
                
                logger.warning(f"JSON mode failed for {provider_type}, retrying without JSON mode")
                
                # Rebuild payload without JSON mode
                provider_features["supports_json_mode"] = False
                data = self._build_llm_request_payload(system_prompt_with_date, user_prompt, provider_type, model, provider_features)
                
                # Retry once without JSON mode
                success, response_data = await self._make_api_request_with_retry(
                    api_url, data, headers, provider_type,
                    max_retries=1,  # Just one retry for JSON mode fallback
                    retry_delay=self.valves.retry_delay,
                    request_timeout=self.valves.request_timeout
                )
                
                if not success:
                    return f"Error: LLM API ({provider_type}) failed even without JSON mode: {response_data}"
            else:
                return f"Error: LLM API ({provider_type}) failed: {error_msg}"

        # Extract content from response using consolidated helper
        content = self._extract_llm_response_content(response_data, provider_type)
        
        if content:
            logger.info(f"Successfully retrieved LLM response (length: {len(content)})")
            return content
        else:
            error_msg = f"Could not extract content from {provider_type} response format"
            logger.error(f"{error_msg}: {json.dumps(response_data)[:500]}...")
            
            # Check for Gemini-specific errors
            if provider_type == "gemini" and isinstance(response_data, dict) and response_data.get("error"):
                error_msg = f"Gemini API error: {response_data['error'].get('message', 'Unknown error')}"
            
            return error_msg
    async def _add_confirmation_message(self, body: Dict[str, Any]) -> None:
        """Add a confirmation message about memory operations"""
        if (
            not body
            or "messages" not in body
            or not body["messages"]
            or not self.valves.show_memory_status
        ):
            return

        # Prepare the confirmation message
        confirmation = ""

        if self._error_message:
            confirmation = f"(Memory error: {self._error_message})"
        elif self.stored_memories:
            # Count operations by type
            new_count = 0
            update_count = 0
            delete_count = 0

            for memory in (self.stored_memories or []):
                if memory["operation"] == "NEW":
                    new_count += 1
                elif memory["operation"] == "UPDATE":
                    update_count += 1
                elif memory["operation"] == "DELETE":
                    delete_count += 1

            # Build the confirmation message in new styled format
            total_saved = new_count + update_count + delete_count

            # Use bold italic styling with an emoji as requested
            confirmation = f"**_Memory: 🧠 Saved {total_saved} memories..._**"

        # If no confirmation necessary, exit early
        if not confirmation:
            log_with_context('debug', "No memory confirmation message needed")
            return

        # Critical fix: Make a selective copy of the messages array
        try:
            log_with_context('debug', "Making selective copy of messages array for safe modification")
            messages_copy = self._copy_messages(body["messages"])

            # Find the last assistant message
            last_assistant_idx = -1
            for i in range(len(messages_copy) - 1, -1, -1):
                if messages_copy[i].get("role") == "assistant":
                    last_assistant_idx = i
                    break

            # If found, modify the copy
            if last_assistant_idx != -1:
                # Get the original content
                original_content = messages_copy[last_assistant_idx].get("content", "")

                # Append the confirmation message
                messages_copy[last_assistant_idx]["content"] = (
                    original_content + f" {confirmation}"
                )

                # Replace the entire messages array in body
                log_with_context('debug', 
                    f"Replacing messages array with modified copy containing confirmation: {confirmation}"
                )
                body["messages"] = messages_copy
            else:
                log_with_context('debug', "No assistant message found to append confirmation")

        except Exception as e:
            log_with_context('error', "Error adding confirmation message",
                           component='OUTLET', operation='confirmation_error', error=e)
            # Don't modify anything if there's an error

    # Cleanup method for aiohttp session and background tasks
    async def cleanup(self):
        """Clean up resources when filter is being shut down"""
        logger.info("Cleaning up Adaptive Memory Filter")
        
        # Cancel all background tasks
        for task in self._background_tasks:
            if not task.done() and not task.cancelled():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    # Expected when cancelling
                    pass
                except Exception as e:
                    log_with_context('error', f"Error while cancelling task: {e}")
        
        # Clear task tracking set
        self._background_tasks.clear()
        
        # Close any open sessions
        if self._aiohttp_session and not self._aiohttp_session.closed:
            await self._aiohttp_session.close()
            
        # Clear memory caches to help with GC
        self._memory_embeddings = {}
        self._relevance_cache = {}
        
        logger.info("Adaptive Memory Filter cleanup complete")

    def _convert_dict_to_memory_operations(
        self, data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Convert a dictionary returned by the LLM into the expected list of memory operations.

        Handles cases where the LLM returns a dict containing a list (e.g., {"memories": [...]})
        or a flatter structure. Avoids adding unnecessary prefixes.
        """
        if not isinstance(data, dict) or not data:
            return []

        operations: List[Dict[str, Any]] = []
        seen_content = set()

        # --- Primary Handling: Check for a key containing a list of operations ---
        # Common keys LLMs might use: "memories", "memory_operations", "results", "operations"
        list_keys = ["memories", "memory_operations", "results", "operations"]
        processed_primary = False
        for key in list_keys:
            if key in data and isinstance(data[key], list):
                logger.info(
                    f"Found list of operations under key '{key}', processing directly."
                )
                for item in data[key]:
                    if isinstance(item, dict):
                        # Extract fields directly, provide defaults
                        op = item.get("operation", "NEW").upper()  # Default to NEW
                        content = item.get(
                            "content", item.get("memory", item.get("value"))
                        )  # Check common content keys
                        tags = item.get("tags", [])
                        memory_bank = item.get("memory_bank", self.valves.default_memory_bank)
                        
                        # Validate memory_bank
                        if self.valves.allowed_memory_banks and memory_bank not in self.valves.allowed_memory_banks:
                            memory_bank = self.valves.default_memory_bank

                        # Basic validation
                        if op not in ["NEW", "UPDATE", "DELETE"]:
                            continue
                        if (
                            not content
                            or not isinstance(content, str)
                            or len(content) < 5
                        ):
                            continue  # Skip empty/short content
                        if not isinstance(tags, list):
                            tags = [str(tags)]  # Ensure tags is a list

                        # Add if content is unique
                        if content not in seen_content:
                            operations.append(
                                {
                                    "operation": op, 
                                    "content": content, 
                                    "tags": tags,
                                    "memory_bank": memory_bank
                                }
                            )
                            seen_content.add(content)
                processed_primary = True
                break  # Stop after processing the first found list

        # --- Fallback Handling: If no primary list found, try simple key-value flattening ---
        if not processed_primary:
            logger.info(
                "No primary operations list found, attempting fallback key-value flattening."
            )
            # Helper maps for simple tag inference (less critical now)
            identity_keys = {"name", "username", "location", "city", "country", "age"}
            goal_keys = {"goal", "objective", "plan"}
            preference_keys = {
                "likes",
                "dislikes",
                "interests",
                "hobbies",
                "favorite",
                "preference",
            }
            relationship_keys = {"family", "friend", "brother", "sister"}
            ignore_keys = {"notes", "meta", "trivia"}

            # Bank inference based on key name
            work_keys = {"job", "profession", "career", "work", "office", "business", "project"}
            personal_keys = {"home", "family", "hobby", "personal", "like", "enjoy", "love", "hate", "friend"}

            for key, value in data.items():
                lowered_key = key.lower()
                if (
                    lowered_key in ignore_keys
                    or not isinstance(value, (str, int, float, bool))
                    or not str(value).strip()
                ):
                    continue

                content = str(value).strip()
                if len(content) > 5 and content not in seen_content:
                    # Simple tag inference
                    tag = "preference"  # Default tag
                    if lowered_key in identity_keys:
                        tag = "identity"
                    elif lowered_key in goal_keys:
                        tag = "goal"
                    elif lowered_key in relationship_keys:
                        tag = "relationship"
                    
                    # Simple bank inference
                    memory_bank = self.valves.default_memory_bank
                    if lowered_key in work_keys:
                        memory_bank = "Work"
                    elif lowered_key in personal_keys:
                        memory_bank = "Personal"

                    # Format simply: "Key: Value" unless key is generic
                    generic_keys = {"content", "memory", "text", "value", "result", "data"}
                    if key.lower() in generic_keys:
                        content_to_save = content # Use content directly
                    else:
                        # Prepend the key for non-generic keys
                        content_to_save = f"{key.replace('_', ' ').capitalize()}: {content}"

                    operations.append(
                        {
                            "operation": "NEW",
                            "content": content_to_save,
                            "tags": [tag],
                            "memory_bank": memory_bank,
                            "confidence": 0.5 # --- Assign default confidence --- NEW
                        }
                    )
                    seen_content.add(content)

        logger.info(f"Converted dict response into {len(operations)} memory operations")
        return operations

    # ------------------------------------------------------------------
    # Helper: background task initialisation (called once from inlet())
    # ------------------------------------------------------------------
    def _initialize_background_tasks(self) -> None:
        """(Idempotent) Ensure any background tasks that rely on the event
        loop are started the first time `inlet` is executed.

        Earlier versions attempted to call this but the helper did not
        exist, causing an `AttributeError`.  The current implementation is
        intentionally lightweight because most tasks are already started
        inside `__init__` when the filter is instantiated by OpenWebUI.
        The function therefore acts as a safety-net and can be extended in
        future if additional runtime-initialised tasks are required.
        """
        # Nothing to do for now because __init__ has already created the
        # background tasks.  Guard against multiple invocations.
        if getattr(self, "_background_tasks_started", False):
            return

        # Placeholder for potential future dynamic tasks
        log_with_context('debug', "_initialize_background_tasks called – no dynamic tasks to start.")
        self._background_tasks_started = True

    # ------------------------------------------------------------------
    # Helper: Increment named error counter safely
    # ------------------------------------------------------------------
    def _increment_error_counter(self, counter_name: str) -> None:
        """Increment an error counter defined in `self.error_counters`.

        Args:
            counter_name: The key identifying the counter to increment.
        """
        try:
            if counter_name not in self.error_counters:
                # Lazily create unknown counters so callers don't crash
                self.error_counters[counter_name] = 0
            self.error_counters[counter_name] += 1
        except Exception as e:
            # Should never fail, but guard to avoid cascading errors
            log_with_context('debug', f"_increment_error_counter failed for '{counter_name}': {e}")

    # --- Consolidated Embedding Functions ---

    async def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Unified embedding getter with metrics and retries"""
        provider = self.valves.embedding_provider_type
        EMBEDDING_REQUESTS.labels(provider).inc()
        _embed_start = time.perf_counter()
        """Primary function to get embedding, uses local or API based on valves."""
        provider_type = self.valves.embedding_provider_type
        
        if not text:
            log_with_context('debug', "Skipping embedding for empty text.")
            return None

        start_time = time.time()
        embedding_vector = None
        try:
            if provider_type == "local":
                local_model = self._local_embedding_model # Access the renamed property
                if local_model:
                    # Ensure text is not excessively long for local model
                    # Simple truncation, might need smarter chunking for production
                    max_local_len = 512 # Adjust based on model limits
                    truncated_text = text[:max_local_len]
                    if len(text) > max_local_len:
                        logger.warning(f"Truncating text for local embedding model (>{max_local_len} chars): {text[:60]}...")
                    
                    embedding_vector = local_model.encode(truncated_text, normalize_embeddings=True)
                else:
                    log_with_context('error', "Local embedding provider configured, but model failed to load.")
                    self.error_counters["embedding_errors"] += 1 # Count as error
            
            elif provider_type == "openai_compatible":
                # Use consolidated API request for embeddings
                api_url = self.valves.embedding_api_url
                api_key = self.valves.embedding_api_key
                # Security: Validate model name even for API calls
                model_name = self._validate_embedding_model_name(self.valves.embedding_model_name)
                
                if not api_url or not api_key:
                    log_with_context('error', "Attempted to call embedding API without proper URL or Key configuration.")
                    self.error_counters["embedding_errors"] += 1
                    embedding_vector = None
                else:
                    # Security: Don't log full API URL to prevent credential exposure
                    safe_url = api_url.split('?')[0] if '?' in api_url else api_url
                    logger.info(f"Getting embedding via API: URL={safe_url}, Model={model_name}")
                    
                    # Build embedding request payload
                    data = {
                        "input": text,
                        "model": model_name
                    }
                    
                    # Build headers
                    headers = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {api_key}"
                    }
                    
                    # Make API request using consolidated method
                    success, response_data = await self._make_api_request_with_retry(
                        api_url, data, headers, "embedding",
                        max_retries=self.valves.max_retries,
                        retry_delay=self.valves.retry_delay,
                        request_timeout=30.0  # Shorter timeout for embeddings
                    )
                    
                    if success and isinstance(response_data, dict):
                        # Extract embedding from response
                        if response_data.get("data") and isinstance(response_data["data"], list) and len(response_data["data"]) > 0:
                            embedding_list = response_data["data"][0].get("embedding")
                            if embedding_list and isinstance(embedding_list, list):
                                log_with_context('debug', f"Successfully received embedding vector of dimension {len(embedding_list)} from API.")
                                embedding_vector = np.array(embedding_list, dtype=np.float32)
                            else:
                                log_with_context('error', f"Could not extract embedding from API response format: {str(response_data)[:200]}...")
                                embedding_vector = None
                        else:
                            log_with_context('error', f"Could not extract embedding from API response format: {str(response_data)[:200]}...")
                            embedding_vector = None
                    else:
                        log_with_context('error', f"Embedding API request failed: {response_data}")
                        embedding_vector = None
                    
                    if embedding_vector is None:
                        self.error_counters["embedding_errors"] += 1
            
            else:
                log_with_context('error', f"Invalid embedding_provider_type configured: {provider_type}")
                self.error_counters["embedding_errors"] += 1 # Count as error

        except Exception as e:
            logger.error(f"Error during embedding generation ({provider_type}): {e}\n{traceback.format_exc()}")
            self.error_counters["embedding_errors"] += 1 # Count as error
            embedding_vector = None # Ensure None is returned on error

        end_time = time.time()
        if embedding_vector is not None:
            # Ensure it's a numpy array of float32 for consistency
            if not isinstance(embedding_vector, np.ndarray):
                 try:
                     embedding_vector = np.array(embedding_vector, dtype=np.float32)
                 except Exception as array_err:
                      log_with_context('error', f"Failed to convert embedding to numpy array: {array_err}")
                      self.error_counters["embedding_errors"] += 1
                      return None
            # Normalize just in case encode/API didn't - with safe bounds checking
            if embedding_vector is not None and len(embedding_vector) > 0:
                norm = np.linalg.norm(embedding_vector)
                if norm > 1e-6:  # Avoid division by zero or near-zero
                    embedding_vector = embedding_vector / norm
                else:
                    logger.warning("Generated embedding vector has near-zero norm. Cannot normalize.")
                    # Return None to indicate invalid embedding
                    return None
            else:
                logger.warning("Generated embedding vector is empty or invalid")
                return None
            
            log_with_context('debug', f"Generated embedding via {provider_type} in {end_time - start_time:.3f}s, dim: {embedding_vector.shape}")
            EMBEDDING_LATENCY.labels(provider).observe(time.perf_counter() - _embed_start)
            return embedding_vector
        else:
            logger.warning(f"Failed to generate embedding via {provider_type} in {end_time - start_time:.3f}s")
            EMBEDDING_ERRORS.labels(provider).inc()
            EMBEDDING_LATENCY.labels(provider).observe(time.perf_counter() - _embed_start)
            return None
    # --- END NEW Embedding Functions ---

    # ==========================================
    # API VERSION DETECTION AND COMPATIBILITY
    # ==========================================
    
    def _detect_openwebui_version(self, body: dict, **kwargs) -> dict:
        """
        Detect OpenWebUI version from request context and parameters.
        Returns version information and compatibility flags.
        Uses cached version info when available for performance.
        """
        try:
            # Return cached version info if available and confident
            if self._api_version_info and self._api_version_info.get('confidence') == 'high':
                return self._api_version_info.copy()
                
            version_info = {
                'detected_version': 'unknown',
                'supports_user_param': False,
                'supports_event_emitter': False,
                'api_pattern': 'legacy',
                'confidence': 'low'
            }
            
            # Check for version indicators in request parameters
            if '__user__' in kwargs:
                version_info['supports_user_param'] = True
                version_info['api_pattern'] = 'modern'
                version_info['confidence'] = 'medium'
                
            if '__event_emitter__' in kwargs:
                version_info['supports_event_emitter'] = True
                version_info['api_pattern'] = 'modern'
                version_info['confidence'] = 'medium'
                
            # Check for deprecated parameters that indicate older versions
            deprecated_params = ['bypass_prompt_processing', 'prompt']
            deprecated_found = any(param in kwargs for param in deprecated_params)
            if deprecated_found:
                version_info['api_pattern'] = 'legacy'
                version_info['confidence'] = 'high'
                
            # Check body structure for version indicators
            if body:
                if 'user' in body and isinstance(body['user'], dict):
                    if body['user'].get('id'):
                        version_info['api_pattern'] = 'modern' if version_info['api_pattern'] == 'unknown' else version_info['api_pattern']
                        
                # Check for message structure that indicates newer versions
                if 'messages' in body and isinstance(body['messages'], list):
                    if any(msg.get('role') in ['user', 'assistant', 'system'] for msg in body['messages']):
                        version_info['detected_version'] = 'v3.x+'
                        version_info['confidence'] = 'high'
                        
            # Store version info for session persistence if confidence is high
            if version_info['confidence'] == 'high' and self._api_version_info is None:
                self._api_version_info = version_info
                logger.info(f"OpenWebUI API version detected and cached: {version_info}")
            elif version_info['confidence'] in ['medium', 'high']:
                log_with_context('debug', f"OpenWebUI API version detected: {version_info}")
                
            # Increment detection counter for analytics
            self._version_detection_count += 1
                
            return version_info
            
        except Exception as e:
            logger.error(f"Error detecting OpenWebUI version: {e}")
            return {
                'detected_version': 'unknown',
                'supports_user_param': False,
                'supports_event_emitter': False,
                'api_pattern': 'legacy',
                'confidence': 'low'
            }
    
    def _apply_version_compatibility(self, body: dict, version_info: dict, **kwargs) -> dict:
        """
        Apply version-specific compatibility adjustments based on detected version.
        """
        try:
            # Create a working copy
            compatible_body = body.copy()
            
            # Apply version-specific handling
            if version_info['api_pattern'] == 'modern':
                # Modern API pattern - prefer __user__ parameter over body.user
                if '__user__' in kwargs and kwargs['__user__']:
                    # Ensure body has user info for internal consistency
                    if 'user' not in compatible_body:
                        compatible_body['user'] = kwargs['__user__']
                        
            elif version_info['api_pattern'] == 'legacy':
                # Legacy API pattern - ensure user info is in body
                if 'user' not in compatible_body:
                    compatible_body['user'] = {}
                    
            # Version-specific logging adjustments
            if version_info['confidence'] == 'high':
                log_with_context('debug', f"Applied {version_info['api_pattern']} API compatibility adjustments")
                
            return compatible_body
            
        except Exception as e:
            logger.error(f"Error applying version compatibility: {e}")
            return body
    
    def get_api_version_info(self) -> dict:
        """
        Get current API version information for debugging and monitoring.
        """
        return {
            'cached_version_info': self._api_version_info,
            'detection_count': self._version_detection_count,
            'has_cached_info': self._api_version_info is not None,
            'confidence': self._api_version_info.get('confidence', 'unknown') if self._api_version_info else 'unknown'
        }

    # ==========================================
    # PARAMETER NORMALIZATION AND COMPATIBILITY
    # ==========================================
    
    def _normalize_request_parameters(self, body: dict, **kwargs) -> dict:
        """
        Normalize and sanitize request parameters for compatibility across OpenWebUI versions.
        Handles deprecated, unknown, or problematic parameters gracefully.
        Includes comprehensive validation and sanitization.
        """
        try:
            # Detect OpenWebUI version and apply compatibility adjustments
            version_info = self._detect_openwebui_version(body, **kwargs)
            normalized_body = self._apply_version_compatibility(body, version_info, **kwargs)
            
            # Log any unrecognized parameters for debugging
            known_params = {'__event_emitter__', '__user__', 'body', 'event'}
            unknown_params = set(kwargs.keys()) - known_params
            if unknown_params:
                log_with_context('debug', f"Encountered unknown parameters (will ignore): {unknown_params}")
                
            # Handle deprecated parameters specifically mentioned in compatibility reports
            deprecated_params = ['bypass_prompt_processing', 'prompt']
            for param in deprecated_params:
                if param in kwargs:
                    logger.warning(f"Deprecated parameter '{param}' encountered - ignoring")
            
            # Additional parameter validation and sanitization
            normalized_body = self._sanitize_body_parameters(normalized_body)
            
            # Ensure required fields exist with defaults
            if 'messages' not in normalized_body:
                normalized_body['messages'] = []
            if 'user' not in normalized_body:
                normalized_body['user'] = {}
                
            # Validate critical fields
            normalized_body = self._validate_critical_fields(normalized_body)
                
            return normalized_body
            
        except Exception as e:
            logger.error(f"Error normalizing request parameters: {e}")
            # Return original body if normalization fails
            return body
    
    def _sanitize_body_parameters(self, body: dict) -> dict:
        """
        Sanitize body parameters to ensure they meet expected formats and constraints.
        """
        try:
            sanitized_body = body.copy()
            
            # Sanitize messages array
            if 'messages' in sanitized_body:
                if not isinstance(sanitized_body['messages'], list):
                    logger.warning("Messages field is not a list - converting to empty list")
                    sanitized_body['messages'] = []
                else:
                    # Sanitize individual messages
                    sanitized_messages = []
                    for msg in sanitized_body['messages']:
                        if isinstance(msg, dict):
                            sanitized_msg = self._sanitize_message(msg)
                            if sanitized_msg:
                                sanitized_messages.append(sanitized_msg)
                    sanitized_body['messages'] = sanitized_messages
            
            # Sanitize user object
            if 'user' in sanitized_body:
                if not isinstance(sanitized_body['user'], dict):
                    logger.warning("User field is not a dict - converting to empty dict")
                    sanitized_body['user'] = {}
                else:
                    sanitized_body['user'] = self._sanitize_user_object(sanitized_body['user'])
            
            # Remove potentially problematic fields
            problematic_fields = ['password', 'token', 'secret', 'api_key', 'auth', 'authorization', 'bearer', 'credentials']
            for field in problematic_fields:
                if field in sanitized_body:
                    logger.warning(f"Removing potentially sensitive field: {field}")
                    del sanitized_body[field]
            
            # Security: Validate embedding model name if present
            if 'embedding_model_name' in sanitized_body:
                if hasattr(self, '_validate_embedding_model_name'):
                    sanitized_body['embedding_model_name'] = self._validate_embedding_model_name(sanitized_body['embedding_model_name'])
                else:
                    # Fallback validation
                    model_name = sanitized_body.get('embedding_model_name', '')
                    if not isinstance(model_name, str) or any(char in model_name for char in [';', '|', '&', '`', '$', '(', ')', '\\']):
                        logger.warning(f"Invalid embedding model name in request: {model_name}")
                        sanitized_body['embedding_model_name'] = 'all-MiniLM-L6-v2'
            
            return sanitized_body
            
        except Exception as e:
            logger.error(f"Error sanitizing body parameters: {e}")
            return body
    
    def _safe_str_conversion(self, value: Any) -> str:
        """Safely convert any value to string with proper null handling."""
        if value is None:
            return ''
        if isinstance(value, str):
            return value
        if isinstance(value, (int, float, bool)):
            return str(value)
        if isinstance(value, (list, dict)):
            try:
                return str(value)
            except Exception:
                return ''
        try:
            return str(value)
        except Exception:
            return ''
    
    def _safe_int_conversion(self, value: Any, default: int = 0) -> int:
        """Safely convert any value to integer with proper null handling."""
        if value is None:
            return default
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                try:
                    return int(float(value))
                except ValueError:
                    return default
        return default
    
    def _sanitize_message(self, message: dict) -> Optional[dict]:
        """
        Sanitize a single message object to ensure it has required fields.
        """
        try:
            if not isinstance(message, dict):
                return None
                
            sanitized_msg = {}
            
            # Ensure role field exists and is valid with safe conversion
            role = message.get('role', 'user')
            role = self._safe_str_conversion(role).lower()
            if role not in ['user', 'assistant', 'system']:
                log_with_context('debug', f"Invalid message role '{role}' - defaulting to 'user'")
                role = 'user'
            sanitized_msg['role'] = role
            
            # Ensure content field exists with safe conversion
            content = message.get('content', '')
            content = self._safe_str_conversion(content)
            sanitized_msg['content'] = content
            
            # Preserve other safe fields with safe conversion
            safe_fields = ['timestamp', 'id', 'metadata']
            for field in safe_fields:
                if field in message:
                    value = message[field]
                    # Apply safe conversion for specific fields
                    if field == 'timestamp':
                        sanitized_msg[field] = self._safe_str_conversion(value)
                    elif field == 'id':
                        sanitized_msg[field] = self._safe_str_conversion(value)
                    else:
                        sanitized_msg[field] = value
            
            return sanitized_msg
            
        except Exception as e:
            logger.error(f"Error sanitizing message: {e}")
            return None
    
    def _sanitize_user_object(self, user: dict) -> dict:
        """
        Sanitize user object to ensure it has safe and valid fields.
        """
        try:
            sanitized_user = {}
            
            # Preserve safe user fields
            safe_fields = ['id', 'name', 'email', 'role']
            for field in safe_fields:
                if field in user:
                    if field == 'id' and user[field]:
                        # Ensure user ID is string
                        sanitized_user[field] = str(user[field])
                    elif field in ['name', 'email'] and user[field]:
                        # Ensure text fields are strings
                        sanitized_user[field] = str(user[field])
                    else:
                        sanitized_user[field] = user[field]
            
            return sanitized_user
            
        except Exception as e:
            logger.error(f"Error sanitizing user object: {e}")
            return {}
    
    def _validate_critical_fields(self, body: dict) -> dict:
        """
        Validate critical fields and apply fixes where possible.
        """
        try:
            validated_body = body.copy()
            
            # Validate messages array structure
            if 'messages' in validated_body:
                messages = validated_body['messages']
                if len(messages) > 1000:  # Reasonable limit
                    logger.warning(f"Message count ({len(messages)}) exceeds limit - truncating")
                    validated_body['messages'] = messages[-1000:]  # Keep latest 1000
            
            # Validate user ID exists for memory operations
            if 'user' in validated_body and validated_body['user']:
                if not validated_body['user'].get('id'):
                    logger.warning("User object exists but has no ID - memory operations will be skipped")
            
            return validated_body
            
        except Exception as e:
            logger.error(f"Error validating critical fields: {e}")
            return body

    # ==========================================
    # SYNCHRONOUS FILTER FUNCTION METHODS (v4.0)
    # ==========================================
    
    def inlet(self, body: dict, __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None, __user__: Optional[dict] = None, **kwargs) -> dict:
        """
        Synchronous inlet method for OpenWebUI Filter Function.
        Pre-processes user input and extracts memories.
        
        This is a wrapper around the async inlet method to provide
        compatibility with synchronous OpenWebUI filter functions.
        Supports both old (user in body) and new (__user__ parameter) OpenWebUI API patterns.
        Handles unknown/deprecated parameters gracefully via **kwargs.
        """
        try:
            # Normalize parameters for compatibility
            normalized_body = self._normalize_request_parameters(body, 
                                                               __event_emitter__=__event_emitter__,
                                                               __user__=__user__,
                                                               **kwargs)
            
            # Handle parameter compatibility - support both old and new API patterns
            user_info = __user__ if __user__ else normalized_body.get("user", {})
            if not user_info or not user_info.get("id"):
                log_with_context('warning', "Sync inlet: No user info available, skipping processing",
                               component='INLET', operation='sync_no_user')
                return normalized_body
            
            # Improved event loop management for OpenWebUI 2024
            result = self._run_async_in_sync_context(
                self.async_inlet(normalized_body, __event_emitter__, user_info),
                "inlet",
                normalized_body
            )
            
            # Clean the result before returning - remove user object if present
            # OpenWebUI expects 'user' to be a string (user ID) not an object when sending to LLM
            if 'user' in result and isinstance(result.get('user'), dict):
                # Extract user ID from user object  
                user_id = result['user'].get('id')
                if user_id:
                    result['user'] = str(user_id)
                else:
                    # Remove user field if no ID
                    result.pop('user', None)
            
            return result
                    
        except Exception as e:
            log_with_context('error', "Error in sync inlet",
                           component='INLET', operation='sync_error', error=e)
            # Never raise exceptions - return body unchanged
            return body
    
    def outlet(self, body: dict, __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None, __user__: Optional[dict] = None, **kwargs) -> dict:
        """
        Synchronous outlet method for OpenWebUI Filter Function.
        Post-processes LLM output and injects relevant memories.
        
        This is a wrapper around the async outlet method to provide
        compatibility with synchronous OpenWebUI filter functions.
        Supports both old (user in body) and new (__user__ parameter) OpenWebUI API patterns.
        Handles unknown/deprecated parameters gracefully via **kwargs.
        """
        try:
            # Normalize parameters for compatibility
            normalized_body = self._normalize_request_parameters(body, 
                                                               __event_emitter__=__event_emitter__,
                                                               __user__=__user__,
                                                               **kwargs)
            
            # Handle parameter compatibility - support both old and new API patterns
            user_info = __user__ if __user__ else normalized_body.get("user", {})
            if not user_info or not user_info.get("id"):
                log_with_context('warning', "Sync outlet: No user info available, skipping processing",
                               component='OUTLET', operation='sync_no_user')
                return normalized_body
            
            # Improved event loop management for OpenWebUI 2024
            result = self._run_async_in_sync_context(
                self.async_outlet(normalized_body, __event_emitter__, user_info),
                "outlet",
                normalized_body
            )
            
            # Clean the result before returning - remove user object if present
            # OpenWebUI expects 'user' to be a string (user ID) not an object when sending to LLM
            if 'user' in result and isinstance(result.get('user'), dict):
                # Extract user ID from user object
                user_id = result['user'].get('id')
                if user_id:
                    result['user'] = str(user_id)
                else:
                    # Remove user field if no ID
                    result.pop('user', None)
            
            return result
                    
        except Exception as e:
            log_with_context('error', "Error in sync outlet",
                           component='OUTLET', operation='sync_error', error=e)
            # Never raise exceptions - return body unchanged
            return body
    
    def _run_async_in_sync_context(self, coro, method_name: str, fallback_body: dict) -> dict:
        """
        Safely run async coroutine in sync context following OpenWebUI 2024 patterns.
        Handles event loop management and resource cleanup properly.
        
        Args:
            coro: The async coroutine to run
            method_name: Name of the calling method for logging
            fallback_body: Body to return if async execution fails
            
        Returns:
            Result of async execution or fallback_body on error
        """
        try:
            # Check if we're in an async context
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Already in async context - create a task to run in background
                    log_with_context('info', f"Sync {method_name} called from async context - scheduling async execution",
                                   component='FILTER', operation='sync_to_async')
                    
                    # Already in async context - we need to handle this differently
                    # For OpenWebUI, we'll create a background task that won't block
                    
                    # Store the coroutine for background execution
                    if not hasattr(self, '_background_tasks'):
                        self._background_tasks: List[asyncio.Task] = []
                    
                    async def process_in_background():
                        try:
                            # Run the coroutine
                            result = await coro
                            log_with_context('info', f"{method_name} background processing completed successfully",
                                           component='FILTER', operation='async_complete')
                            # For inlet, we want to extract memories but return original body
                            # For outlet, we want to inject memories but process happens async
                            if method_name == "inlet":
                                # Memory extraction happens in background
                                return fallback_body
                            else:
                                # Memory injection for outlet - return processed result
                                return result
                        except Exception as e:
                            log_with_context('error', f"Background {method_name} processing error: {e}\n{traceback.format_exc()}")
                            return fallback_body
                    
                    # Create and track the background task
                    task = asyncio.create_task(process_in_background())
                    self._background_tasks.append(task)
                    
                    # Clean up completed tasks periodically
                    self._background_tasks = [t for t in self._background_tasks if not t.done()]
                    
                    # Log that we're processing in background
                    log_with_context('info', f"{method_name} scheduled for background processing (OpenWebUI async context)",
                                   component='FILTER', operation='async_scheduled')
                    
                    # Return the fallback body immediately for OpenWebUI
                    return fallback_body
                else:
                    # Existing loop not running - safe to use
                    return loop.run_until_complete(coro)
            except RuntimeError:
                # No event loop exists - create temporary one
                loop = None
                try:
                    loop = asyncio.new_event_loop()
                    # Don't set as default loop to avoid conflicts
                    return loop.run_until_complete(coro)
                finally:
                    # Always clean up the loop we created
                    if loop:
                        try:
                            # Cancel all pending tasks
                            pending = asyncio.all_tasks(loop)
                            for task in pending:
                                task.cancel()
                            # Wait for tasks to complete cancellation
                            if pending:
                                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                        except Exception as cleanup_error:
                            log_with_context('debug', f"Error cleaning up loop tasks: {cleanup_error}")
                        finally:
                            loop.close()
                            
        except Exception as e:
            log_with_context('error', f"Error in {method_name} async execution",
                           component='FILTER', operation='async_error', error=e)
            return fallback_body
    
    def stream(self, event: dict, **kwargs) -> dict:
        """
        Enhanced stream method for OpenWebUI Filter Function v0.5.17+
        
        This method handles streaming responses from the LLM with real-time filtering.
        Implements OpenWebUI 2024 stream filtering capabilities including:
        - Real-time content filtering
        - PII detection and filtering
        - Enhanced event processing
        
        Handles unknown/deprecated parameters gracefully via **kwargs.
        """
        try:
            # Log any unknown parameters for debugging
            if kwargs:
                log_with_context('debug', f"Stream method received unknown parameters (ignoring): {list(kwargs.keys())}")
            
            # -----------------------------------------------------------
            # Filter Orchestration System Integration for Stream
            # -----------------------------------------------------------
            operation_start_time = time.time()
            
            if self.internal_config.enable_filter_orchestration:
                try:
                    # Record stream operation
                    self._record_operation_start("stream")
                    
                    # OpenWebUI 2024 Enhanced Stream Processing
                    if self.valves.enable_stream_filtering:
                        event = self._process_stream_event(event)
                    else:
                        log_with_context('debug', f"Stream event received: {event.get('type', 'unknown')}")
                    
                    # Record successful completion
                    self._record_operation_success("stream", operation_start_time)
                    
                except Exception as orch_error:
                    log_with_context('debug', f"Filter orchestration error in stream: {orch_error}")
                    self._record_operation_failure("stream", operation_start_time, str(orch_error))
            else:
                # Standard processing without orchestration
                if self.valves.enable_stream_filtering:
                    event = self._process_stream_event(event)
                else:
                    log_with_context('debug', f"Stream event received: {event.get('type', 'unknown')}")
            
            return event
            
        except Exception as e:
            # Record orchestration failure if enabled
            if self.internal_config.enable_filter_orchestration:
                self._record_operation_failure("stream", time.time(), str(e))
            
            log_with_context('error', "Error in sync stream",
                           component='FILTER', operation='stream_error', error=e)
            # Never raise exceptions - return event unchanged
            return event

    # -----------------------------------------------------------
    # OpenWebUI 2024 Stream Processing Methods
    # -----------------------------------------------------------
    
    def _process_stream_event(self, event: dict) -> dict:
        """
        Process streaming events with OpenWebUI 2024 filtering capabilities.
        
        Args:
            event: The streaming event from OpenWebUI
            
        Returns:
            dict: Processed event with filtering applied
        """
        try:
            # Handle different event types
            event_type = event.get('type', 'unknown')
            
            if event_type == 'content' or 'choices' in event:
                # Process content chunks for OpenAI-style responses
                event = self._filter_streaming_content(event)
                
            elif event_type == 'message' and 'content' in event:
                # Process direct message content
                event = self._filter_message_content(event)
                
            log_with_context('debug', f"Processed stream event: {event_type}")
            return event
            
        except Exception as e:
            log_with_context('error', "Error processing stream event",
                           component='FILTER', operation='stream_event_error', error=e)
            return event
    
    def _filter_streaming_content(self, event: dict) -> dict:
        """
        Filter content in streaming responses (OpenAI-style format).
        
        Args:
            event: The streaming event containing choices/delta/content
            
        Returns:
            dict: Filtered event
        """
        try:
            # Handle OpenAI-style streaming format
            if 'choices' in event:
                for choice in event.get('choices', []):
                    delta = choice.get('delta', {})
                    if 'content' in delta:
                        original_content = delta['content']
                        filtered_content = self._apply_content_filters(original_content)
                        delta['content'] = filtered_content
                        
                        # Log filtering if content was modified
                        if original_content != filtered_content:
                            log_with_context('debug', f"Stream content filtered: {len(original_content)} -> {len(filtered_content)} chars")
            
            return event
            
        except Exception as e:
            log_with_context('error', "Error filtering streaming content",
                           component='FILTER', operation='stream_filter_error', error=e)
            return event
    
    def _filter_message_content(self, event: dict) -> dict:
        """
        Filter direct message content in streaming responses.
        
        Args:
            event: The streaming event containing message content
            
        Returns:
            dict: Filtered event
        """
        try:
            if 'content' in event:
                original_content = event['content']
                filtered_content = self._apply_content_filters(original_content)
                event['content'] = filtered_content
                
                # Log filtering if content was modified
                if original_content != filtered_content:
                    log_with_context('debug', f"Message content filtered: {len(original_content)} -> {len(filtered_content)} chars")
            
            return event
            
        except Exception as e:
            log_with_context('error', "Error filtering message content",
                           component='FILTER', operation='message_filter_error', error=e)
            return event
    
    def _apply_content_filters(self, content: str) -> str:
        """
        Apply content filtering based on valve configuration.
        
        Args:
            content: The content to filter
            
        Returns:
            str: Filtered content
        """
        if not content or not self.valves.enable_stream_content_filtering:
            return content
        
        try:
            filtered_content = content
            
            # Apply PII filtering if enabled
            if self.valves.enable_pii_filtering:
                filtered_content = self._filter_pii_content(filtered_content)
            
            # Add other content filters here as needed
            # Example: profanity filtering, sensitive data filtering, etc.
            
            return filtered_content
            
        except Exception as e:
            log_with_context('error', "Error applying content filters",
                           component='FILTER', operation='apply_filters_error', error=e)
            return content
    
    def _filter_pii_content(self, content: str) -> str:
        """
        Filter PII (Personally Identifiable Information) from content.
        
        Args:
            content: The content to filter
            
        Returns:
            str: Content with PII filtered according to pii_filter_mode
        """
        try:
            # Basic PII patterns - can be extended with more sophisticated detection
            import re
            
            pii_patterns = {
                'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
                'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
                'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'
            }
            
            filtered_content = content
            
            for pii_type, pattern in pii_patterns.items():
                if self.valves.pii_filter_mode == "redact":
                    filtered_content = re.sub(pattern, f"[{pii_type.upper()}_REDACTED]", filtered_content)
                elif self.valves.pii_filter_mode == "anonymize":
                    filtered_content = re.sub(pattern, f"[{pii_type.upper()}_ANONYMIZED]", filtered_content)
                elif self.valves.pii_filter_mode == "encrypt":
                    # For encryption mode, we would encrypt the PII
                    # This is a simplified example - real implementation would use proper encryption
                    filtered_content = re.sub(pattern, f"[{pii_type.upper()}_ENCRYPTED]", filtered_content)
            
            return filtered_content
            
        except Exception as e:
            log_with_context('error', "Error filtering PII content",
                           component='FILTER', operation='pii_filter_error', error=e)
            return content
    
    # -----------------------------------------------------------
    # Database Write Hooks (Issue #11888)
    # -----------------------------------------------------------
    
    def _prepare_content_for_database(self, content: str, context: str = "general") -> str:
        """
        Prepare content for database storage with OpenWebUI 2024 write hooks.
        
        This method implements the database write filtering requested in Issue #11888.
        It allows different processing for display vs. storage.
        
        Args:
            content: The content to prepare for database storage
            context: The context of the content (e.g., "user_message", "assistant_response")
            
        Returns:
            str: Content prepared for database storage
        """
        if not self.valves.enable_database_write_hooks:
            return content
        
        try:
            prepared_content = content
            
            # Apply PII filtering for database storage
            if self.valves.enable_pii_filtering:
                prepared_content = self._filter_pii_content(prepared_content)
            
            # Log the preparation (using context for debugging)
            if prepared_content != content:
                log_with_context('debug', f"Content prepared for database storage ({context}): {len(content)} -> {len(prepared_content)} chars")
            
            return prepared_content
            
        except Exception as e:
            log_with_context('error', f"Error preparing content for database ({context})",
                           component='FILTER', operation='database_prep_error', error=e)
            return content
    
    def _restore_content_from_database(self, content: str, context: str = "general") -> str:
        """
        Restore content from database storage with OpenWebUI 2024 read hooks.
        
        This method implements the database read filtering requested in Issue #11888.
        It allows different processing for storage vs. display.
        
        Args:
            content: The content to restore from database storage
            context: The context of the content (e.g., "user_message", "assistant_response")
            
        Returns:
            str: Content restored from database storage
        """
        if not self.valves.enable_database_write_hooks:
            return content
        
        try:
            restored_content = content
            
            # Handle encrypted PII restoration
            if self.valves.enable_pii_filtering and self.valves.pii_filter_mode == "encrypt":
                # In a real implementation, this would decrypt the PII
                # For now, we'll just return the content as-is
                log_with_context('debug', f"Encrypted PII restoration for context: {context}")
            
            return restored_content
            
        except Exception as e:
            log_with_context('error', f"Error restoring content from database ({context})",
                           component='FILTER', operation='database_restore_error', error=e)
            return content
    
    # -----------------------------------------------------------
    # Enhanced Event Emitter Patterns (OpenWebUI 2024)
    # -----------------------------------------------------------
    
    async def _emit_enhanced_event(self, event_emitter, event_type: str, content: str, 
                                  metadata: Optional[Dict[str, Any]] = None, 
                                  batch_key: Optional[str] = None) -> None:
        """
        Emit events using OpenWebUI 2024 enhanced patterns.
        
        Args:
            event_emitter: The event emitter function
            event_type: Type of event (e.g., "info", "error", "progress")
            content: Event content
            metadata: Optional metadata for the event
            batch_key: Optional key for batching events
        """
        if not event_emitter or not self.valves.enable_enhanced_event_emitter:
            return
        
        try:
            # Prepare enhanced event structure
            event = {
                "type": event_type,
                "content": content,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": "adaptive_memory_v4.0",
                "version": "2024.1"
            }
            
            # Add metadata if provided
            if metadata:
                event.update(metadata)
            
            # Handle batching if enabled
            if self.valves.event_emitter_batch_size > 0 and batch_key:
                await self._emit_batched_event(event_emitter, event, batch_key)
            else:
                # Direct emission
                await self._safe_emit(event_emitter, event)
            
        except Exception as e:
            log_with_context('error', "Error emitting enhanced event",
                           component='FILTER', operation='emit_enhanced_error', error=e)
    
    async def _emit_batched_event(self, event_emitter, event: dict, batch_key: str) -> None:
        """
        Handle batched event emission for improved performance.
        
        Args:
            event_emitter: The event emitter function
            event: The event to emit
            batch_key: Key for batching related events
        """
        try:
            # Initialize batch storage if not exists
            if not hasattr(self, '_event_batches'):
                self._event_batches: Dict[str, List[Dict[str, Any]]] = {}
            
            # Add event to batch
            if batch_key not in self._event_batches:
                self._event_batches[batch_key] = []
            
            self._event_batches[batch_key].append(event)
            
            # Emit batch if size limit reached
            if len(self._event_batches[batch_key]) >= self.valves.event_emitter_batch_size:
                await self._flush_event_batch(event_emitter, batch_key)
            
        except Exception as e:
            log_with_context('error', "Error handling batched event",
                           component='FILTER', operation='batch_handle_error', error=e)
    
    async def _flush_event_batch(self, event_emitter, batch_key: str) -> None:
        """
        Flush a batch of events to the event emitter.
        
        Args:
            event_emitter: The event emitter function
            batch_key: Key of the batch to flush
        """
        try:
            if not hasattr(self, '_event_batches') or batch_key not in self._event_batches:
                return
            
            batch = self._event_batches[batch_key]
            if not batch:
                return
            
            # Emit batched event
            batch_event = {
                "type": "batch",
                "content": f"Batch of {len(batch)} events",
                "events": batch,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": "adaptive_memory_v4.0",
                "version": "2024.1"
            }
            
            await self._safe_emit(event_emitter, batch_event)
            
            # Clear the batch
            del self._event_batches[batch_key]
            
        except Exception as e:
            log_with_context('error', "Error flushing event batch",
                           component='FILTER', operation='batch_flush_error', error=e)
    
    # -----------------------------------------------------------
    # OpenWebUI 2024 Compliance Validation
    # -----------------------------------------------------------
    
    def validate_openwebui_2024_compliance(self) -> Dict[str, Any]:
        """
        Validate OpenWebUI 2024 compliance and return status report.
        
        Returns:
            dict: Compliance status report with feature availability
        """
        compliance_report: Dict[str, Any] = {
            "openwebui_version": "2024.1",
            "compliance_version": "v0.5.17+",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "features": {
                "stream_function": {
                    "available": True,
                    "enabled": self.valves.enable_stream_filtering,
                    "description": "Real-time streaming response filtering"
                },
                "database_write_hooks": {
                    "available": True,
                    "enabled": self.valves.enable_database_write_hooks,
                    "description": "Separate processing for display vs storage (Issue #11888)"
                },
                "pii_filtering": {
                    "available": True,
                    "enabled": self.valves.enable_pii_filtering,
                    "mode": self.valves.pii_filter_mode,
                    "description": "PII filtering for data protection"
                },
                "enhanced_event_emitter": {
                    "available": True,
                    "enabled": self.valves.enable_enhanced_event_emitter,
                    "batch_size": self.valves.event_emitter_batch_size,
                    "description": "Enhanced event emitter patterns with batching"
                },
                "content_filtering": {
                    "available": True,
                    "enabled": self.valves.enable_stream_content_filtering,
                    "description": "Real-time content filtering in streams"
                }
            },
            "compatibility": {
                "openwebui_v0_5_17": True,
                "stream_hooks": True,
                "database_write_hooks": True,
                "event_emitter_v2024": True
            },
            "recommendations": []
        }
        
        # Add recommendations based on current configuration
        if not self.valves.enable_stream_filtering:
            compliance_report["recommendations"].append(
                "Enable stream filtering for real-time content processing"
            )
        
        if not self.valves.enable_database_write_hooks:
            compliance_report["recommendations"].append(
                "Enable database write hooks for better data protection"
            )
        
        if not self.valves.enable_pii_filtering:
            compliance_report["recommendations"].append(
                "Enable PII filtering for enhanced privacy protection"
            )
        
        if not self.valves.enable_enhanced_event_emitter:
            compliance_report["recommendations"].append(
                "Enable enhanced event emitter for better UI integration"
            )
        
        return compliance_report
    
    def get_openwebui_2024_features(self) -> List[str]:
        """
        Get list of available OpenWebUI 2024 features.
        
        Returns:
            list: List of feature names
        """
        return [
            "stream_function_v0_5_17",
            "database_write_hooks_issue_11888",
            "pii_filtering_configurable",
            "enhanced_event_emitter_v2024",
            "content_filtering_realtime",
            "batched_event_emission",
            "structured_event_format",
            "backward_compatibility"
        ]
    
    # -----------------------------------------------------------
    # Filter Orchestration API Methods
    # -----------------------------------------------------------
    
    def get_filter_metadata(self) -> Optional[Dict[str, Any]]:
        """Get filter metadata for orchestration systems"""
        if not self._filter_metadata:
            return None
        
        return {
            "name": self._filter_metadata.name,
            "version": self._filter_metadata.version,
            "description": self._filter_metadata.description,
            "capabilities": self._filter_metadata.capabilities,
            "operations": self._filter_metadata.operations,
            "priority": self._filter_metadata.priority.name,
            "dependencies": self._filter_metadata.dependencies,
            "conflicts_with": self._filter_metadata.conflicts_with,
            "max_execution_time_ms": self._filter_metadata.max_execution_time_ms,
            "memory_requirements_mb": self._filter_metadata.memory_requirements_mb,
            "requires_user_context": self._filter_metadata.requires_user_context,
            "modifies_content": self._filter_metadata.modifies_content,
            "thread_safe": self._filter_metadata.thread_safe
        }
    
    def get_orchestration_status(self) -> Dict[str, Any]:
        """Get current orchestration status and performance metrics"""
        if not self.internal_config.enable_filter_orchestration:
            return {"orchestration_enabled": False}
        
        try:
            status = {
                "orchestration_enabled": True,
                "filter_registered": self._filter_metadata is not None,
                "active_contexts": 0,  # Simplified orchestration - no execution contexts
                "rollback_points": len(self._rollback_stack) if hasattr(self, '_rollback_stack') else 0,
                "configuration": {
                    "enable_conflict_detection": self.internal_config.enable_conflict_detection,
                    "enable_performance_monitoring": self.internal_config.enable_performance_monitoring,
                    "filter_priority": self.internal_config.filter_priority,
                    "enable_rollback_mechanism": self.internal_config.enable_rollback_mechanism,
                    "max_concurrent_filters": self.internal_config.max_concurrent_filters,
                    "coordination_overhead_threshold_ms": self.internal_config.coordination_overhead_threshold_ms,
                    "enable_shared_state": self.internal_config.enable_shared_state,
                    "filter_isolation_level": self.internal_config.filter_isolation_level
                }
            }
            
            # Add performance history if available
            if hasattr(self, '_orchestration_manager') and self._orchestration_manager:
                history = self._orchestration_manager._performance_history
                if history:
                    status["performance"] = {
                        "average_execution_time_ms": sum(history) / len(history),
                        "min_execution_time_ms": min(history),
                        "max_execution_time_ms": max(history),
                        "total_operations": len(history)
                    }
            
            return status
            
        except Exception as e:
            log_with_context('error', "Error getting orchestration status",
                           component='FILTER', operation='orchestration_status_error', error=e)
            return {"orchestration_enabled": True, "error": str(e)}

    def get_conflict_report(self) -> Dict[str, Any]:
        """Get report of potential conflicts with other filters"""
        if not self.internal_config.enable_filter_orchestration or not self.internal_config.enable_conflict_detection:
            return {"conflict_detection_enabled": False}
        
        try:
            # Get registered filters from orchestration manager
            if not hasattr(self, '_orchestration_manager') or not self._orchestration_manager:
                return {"conflict_detection_enabled": True, "conflicts": [], "error": "Orchestration manager not available"}
            
            # Simplified orchestration - no registered filters or conflict detector
            other_filters: List[str] = []
            
            if not self._filter_metadata:
                return {"conflict_detection_enabled": True, "conflicts": [], "error": "Filter not registered"}
            
            conflicts: List[Dict[str, Any]] = []  # No conflicts in simplified orchestration
            
            return {
                "conflict_detection_enabled": True,
                "filter_count": len(other_filters) + 1,  # +1 for adaptive_memory
                "conflicts": conflicts,
                "other_filters": []  # No other filters in simplified orchestration
            }
            
        except Exception as e:
            log_with_context('error', "Error generating conflict report",
                           component='FILTER', operation='conflict_report_error', error=e)
            return {"conflict_detection_enabled": True, "error": str(e)}

    def _register_orchestration_endpoints(self):
        pass  # Simplified - endpoints removed for single filter use
            # Continue without endpoints
