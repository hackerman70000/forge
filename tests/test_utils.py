from loguru import logger

from forge.utils import setup_logging


class TestSetupLogging:
    def test_setup_logging_removes_existing_handlers(self):
        setup_logging(level="INFO")
        setup_logging(level="DEBUG")

    def test_setup_logging_with_log_file(self, tmp_path):
        log_file = tmp_path / "app.log"
        setup_logging(level="INFO", log_file=log_file)

        logger.info("Test message for file logging")

        assert log_file.exists()

    def test_setup_logging_without_log_file(self):
        setup_logging(level="WARNING")

    def test_setup_logging_different_levels(self):
        for level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            setup_logging(level=level)

    def test_log_file_receives_messages(self, tmp_path):
        log_file = tmp_path / "test.log"
        setup_logging(level="DEBUG", log_file=log_file)

        logger.debug("debug message")
        logger.info("info message")
        logger.warning("warning message")

        import time

        time.sleep(0.05)

        content = log_file.read_text()
        assert "debug message" in content
        assert "info message" in content
        assert "warning message" in content

    def test_log_file_in_nested_directory(self, tmp_path):
        nested = tmp_path / "logs" / "subdir"
        nested.mkdir(parents=True)
        log_file = nested / "forge.log"

        setup_logging(level="INFO", log_file=log_file)
        logger.info("Nested dir test")

        import time

        time.sleep(0.05)

        assert log_file.exists()

    def test_multiple_calls_do_not_duplicate_handlers(self, tmp_path):
        log_file = tmp_path / "test.log"

        setup_logging(level="INFO", log_file=log_file)
        setup_logging(level="INFO", log_file=log_file)
        setup_logging(level="INFO", log_file=log_file)

        logger.info("Only once")

        import time

        time.sleep(0.05)

        content = log_file.read_text()
        assert content.count("Only once") == 1
