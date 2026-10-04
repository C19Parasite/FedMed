"""Privacy-safe logging writing to console and optional log file."""

from __future__ import annotations

import logging
from pathlib import Path
import re
from typing import Optional

# Pattern matching file paths, drive letters, extensions to protect patient confidentiality
_PATH_PATTERN = re.compile(
    r"(?:[A-Za-z]:)?[/\\][^\s,;]+"
    r"|(?:[A-Za-z]:[/\\][^\s,;]+)"
    r"|(?:[~.]?[/\\][^\s,;]+)"
    r"|(?:\S+\.(?:nii(?:\.gz)?|dcm|dicom|png|json|yaml|yml))",
    re.IGNORECASE,
)


def redact_paths(text: str) -> str:
    """Replace path-like substrings with a privacy placeholder.

    Args:
        text: Raw text containing potential filesystem paths.

    Returns:
        String with path elements redacted.
    """
    return _PATH_PATTERN.sub("<redacted>", text)


class PrivacyLogFilter(logging.Filter):
    """Filter that intercepts and redacts filesystem paths from LogRecords."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact_paths(str(record.msg))
        if record.args:
            if isinstance(record.args, dict):
                record.args = {k: _redact_arg(v) for k, v in record.args.items()}
            else:
                record.args = tuple(_redact_arg(arg) for arg in record.args)
        return True


def _redact_arg(value: object) -> object:
    if isinstance(value, str):
        return redact_paths(value)
    return value


def get_logger(
    name: str,
    log_file: Optional[str | Path] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Return a privacy-filtered logger configured with console and optional file output.

    Args:
        name: Logger name, typically ``__name__``.
        log_file: Optional path to write log messages.
        level: Logging level (defaults to INFO).

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    privacy_filter = PrivacyLogFilter()
    if not any(isinstance(f, PrivacyLogFilter) for f in logger.filters):
        logger.addFilter(privacy_filter)

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Attach console stream handler if not already attached
    has_stream_handler = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in logger.handlers
    )
    if not has_stream_handler:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(level)
        stream_handler.setFormatter(formatter)
        stream_handler.addFilter(privacy_filter)
        logger.addHandler(stream_handler)

    # Attach file handler if requested and not already present for this path
    if log_file:
        file_path = Path(log_file).resolve()
        file_path.parent.mkdir(parents=True, exist_ok=True)
        already_attached = any(
            isinstance(h, logging.FileHandler)
            and getattr(h, "baseFilename", None) == str(file_path)
            for h in logger.handlers
        )
        if not already_attached:
            file_handler = logging.FileHandler(str(file_path), encoding="utf-8")
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            file_handler.addFilter(privacy_filter)
            logger.addHandler(file_handler)

    logger.propagate = False
    return logger
