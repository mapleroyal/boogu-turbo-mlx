class BooguTurboMlxError(Exception):
    """Base exception for project-level failures."""


class ComponentNotImplementedError(BooguTurboMlxError, NotImplementedError):
    """Raised when a component is intentionally outside the current runtime scope."""


class BooguTurboGenerationCancelled(BooguTurboMlxError):
    """Raised when generation is cooperatively cancelled."""
