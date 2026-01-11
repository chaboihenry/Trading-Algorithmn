"""
Logging Configuration for RiskLabAI Trading Bot

Provides centralized logging setup with:
- Rotating log files (prevents disk space issues)
- Separate files for different log levels
- Console output for immediate feedback
- Structured format with timestamps and module names

OOP Concepts:
- Uses composition with RotatingFileHandler
- Factory pattern with setup_logging() function
- Separation of concerns (logging config in its own module)
"""

import logging
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pathlib import Path
from datetime import datetime


def setup_logging(
    log_dir: str = "logs",
    max_bytes: int = 10_000_000,  # 10MB per log file
    backup_count: int = 5,  # Keep 5 old log files
    level: int = logging.INFO
):
    """
    Configure rotating log files for the trading bot.

    This function sets up a comprehensive logging system with:
    1. Console output for immediate visibility
    2. Rotating file handler that creates new files at 10MB
    3. Separate error log for critical issues
    4. Timestamped log files for each trading session

    Args:
        log_dir: Directory to store log files (default: "logs")
        max_bytes: Max size per log file before rotation (default: 10MB)
        backup_count: Number of backup files to keep (default: 5)
        level: Logging level (default: INFO)

    Returns:
        Configured root logger

    Example:
        >>> from config.logging_config import setup_logging
        >>> logger = setup_logging()
        >>> logger.info("Trading bot started")
    """
    # Create logs directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True, parents=True)

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove any existing handlers (prevents duplicate logs)
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # =========================================================================
    # CONSOLE HANDLER - Shows INFO and above in terminal
    # =========================================================================
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))

    # =========================================================================
    # MAIN LOG FILE - Rotating file with all logs (DEBUG and above)
    # =========================================================================
    main_log = RotatingFileHandler(
        log_path / "trading_bot.log",
        maxBytes=max_bytes,
        backupCount=backup_count
    )
    main_log.setLevel(logging.DEBUG)
    main_log.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))

    # =========================================================================
    # ERROR LOG FILE - Separate file for errors and warnings
    # =========================================================================
    error_log = RotatingFileHandler(
        log_path / "trading_bot_errors.log",
        maxBytes=max_bytes,
        backupCount=backup_count
    )
    error_log.setLevel(logging.WARNING)
    error_log.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s\n'
        'File: %(pathname)s:%(lineno)d\n'
        'Function: %(funcName)s\n',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))

    # =========================================================================
    # DAILY LOG FILE - Creates a new file each day (for historical analysis)
    # =========================================================================
    daily_log = TimedRotatingFileHandler(
        log_path / "trading_bot_daily.log",
        when='midnight',
        interval=1,
        backupCount=30  # Keep 30 days of logs
    )
    daily_log.setLevel(logging.INFO)
    daily_log.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))

    # Add all handlers to root logger
    root_logger.addHandler(console)
    root_logger.addHandler(main_log)
    root_logger.addHandler(error_log)
    root_logger.addHandler(daily_log)

    # Log the logging configuration
    root_logger.info("=" * 80)
    root_logger.info("LOGGING CONFIGURED")
    root_logger.info("=" * 80)
    root_logger.info(f"Log directory: {log_path.absolute()}")
    root_logger.info(f"Main log: trading_bot.log (rotates at {max_bytes / 1_000_000:.1f}MB)")
    root_logger.info(f"Error log: trading_bot_errors.log")
    root_logger.info(f"Daily log: trading_bot_daily.log (new file each day)")
    root_logger.info(f"Backup count: {backup_count} files")
    root_logger.info("=" * 80)

    return root_logger


def get_session_log_file(log_dir: str = "logs") -> Path:
    """
    Get a timestamped log file for this trading session.

    Creates a unique log file with timestamp for each bot run.
    Useful for debugging specific sessions.

    Args:
        log_dir: Directory to store log files

    Returns:
        Path to session log file

    Example:
        >>> log_file = get_session_log_file()
        >>> print(log_file)  # logs/session_20260111_143052.log
    """
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True, parents=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return log_path / f"session_{timestamp}.log"


def add_session_log_handler(logger: logging.Logger = None) -> Path:
    """
    Add a timestamped session log handler to capture this run.

    Args:
        logger: Logger to add handler to (default: root logger)

    Returns:
        Path to session log file
    """
    if logger is None:
        logger = logging.getLogger()

    session_file = get_session_log_file()

    session_handler = logging.FileHandler(session_file)
    session_handler.setLevel(logging.DEBUG)
    session_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))

    logger.addHandler(session_handler)
    logger.info(f"Session log: {session_file}")

    return session_file


# Convenience function to silence noisy libraries
def silence_library_logs():
    """
    Silence noisy third-party library logs.

    Some libraries (like urllib3, alpaca) log too much.
    This reduces their verbosity.
    """
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('alpaca').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
