class PipelineError(RuntimeError):
    """Base error raised for recoverable pipeline failures."""


class DependencyMissingError(PipelineError):
    """Raised when an optional dependency is required for a stage."""


class SchemaError(PipelineError):
    """Raised when manifest or tensor data violates the public contract."""

