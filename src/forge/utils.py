from pathlib import Path

from loguru import logger
from rich.console import Console


def setup_logging(
    level: str = "INFO",
    log_file: Path | None = None,
) -> None:
    logger.remove()

    console = Console(width=None)
    logger.add(
        lambda msg: console.print(msg, end=""),
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}:{function}:{line}</cyan> | {message}",
        level=level,
        colorize=True,
    )

    if log_file:
        logger.add(
            log_file,
            rotation="10 MB",
            retention="30 days",
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        )
