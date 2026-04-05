import sys
from pathlib import Path

import pytest
from invoke import MockContext, Result

sys.path.insert(0, str(Path(__file__).parent.parent))

from tasks import check, format, lint, setup, test, validate


@pytest.fixture
def ctx():
    return MockContext(run=Result())


class TestTestTask:
    def test_default(self, ctx):
        test(ctx)
        assert ctx.run.called
        cmd = ctx.run.call_args[0][0]
        assert cmd == "uv run pytest tests/ -v"

    def test_with_cov(self, ctx):
        test(ctx, cov=True)
        cmd = ctx.run.call_args[0][0]
        assert "--cov=forge" in cmd
        assert "--cov-report=term-missing" in cmd
        assert "--cov-report=html" not in cmd

    def test_with_html(self, ctx):
        test(ctx, html=True)
        cmd = ctx.run.call_args[0][0]
        assert "--cov=forge" in cmd
        assert "--cov-report=html" in cmd

    def test_with_k_filter(self, ctx):
        test(ctx, k="config")
        cmd = ctx.run.call_args[0][0]
        assert "-k config" in cmd

    def test_all_flags(self, ctx):
        test(ctx, cov=True, html=True, k="data")
        cmd = ctx.run.call_args[0][0]
        assert "-k data" in cmd
        assert "--cov=forge" in cmd
        assert "--cov-report=html" in cmd


class TestLintTask:
    def test_default(self, ctx):
        lint(ctx)
        cmd = ctx.run.call_args[0][0]
        assert cmd == "uv run ruff check ."

    def test_with_fix(self, ctx):
        lint(ctx, fix=True)
        cmd = ctx.run.call_args[0][0]
        assert cmd == "uv run ruff check . --fix"


class TestFormatTask:
    def test_default(self, ctx):
        format(ctx)
        cmd = ctx.run.call_args[0][0]
        assert cmd == "uv run ruff format ."

    def test_with_check(self, ctx):
        format(ctx, check=True)
        cmd = ctx.run.call_args[0][0]
        assert cmd == "uv run ruff format . --check"


class TestCheckTask:
    def test_runs_without_error(self, ctx):
        check(ctx)

    def test_has_lint_and_format_as_pre_tasks(self):
        assert lint in check.pre
        assert format in check.pre


class TestSetupTask:
    def test_default_with_gpu(self, ctx):
        setup(ctx)
        cmd = ctx.run.call_args[0][0]
        assert "--extra dev" in cmd
        assert "--extra training" in cmd
        assert "--extra hub" in cmd

    def test_cpu_only(self, ctx):
        setup(ctx, cpu=True)
        cmd = ctx.run.call_args[0][0]
        assert "--extra dev" in cmd
        assert "--extra training" not in cmd
        assert "--extra hub" not in cmd


class TestValidateTask:
    def test_passes_config_path(self, ctx):
        validate(ctx, config="tasks/vulnllm.yaml")
        cmd = ctx.run.call_args[0][0]
        assert "load_task_config" in cmd
        assert "tasks/vulnllm.yaml" in cmd
