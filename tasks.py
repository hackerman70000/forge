from invoke import Collection, task


@task
def test(c, cov=False, html=False, k=None):
    cmd = "uv run pytest tests/ -v"
    if k:
        cmd += f" -k {k}"
    if cov or html:
        cmd += " --cov=forge --cov-report=term-missing"
    if html:
        cmd += " --cov-report=html"
    c.run(cmd, pty=True)


@task
def lint(c, fix=False):
    cmd = "uv run ruff check ."
    if fix:
        cmd += " --fix"
    c.run(cmd, pty=True)


@task
def format(c, check=False):
    cmd = "uv run ruff format ."
    if check:
        cmd += " --check"
    c.run(cmd, pty=True)


@task(pre=[lint, format])
def check(c):
    test(c, cov=True)


@task
def setup(c, cpu=False):
    extras = "--extra dev"
    if not cpu:
        extras += " --extra training --extra hub"
    c.run(f"uv sync {extras}", pty=True)


@task
def validate(c, config):
    c.run(
        f"uv run python -c \"from forge.config import load_task_config; c = load_task_config('{config}'); print(c.model_dump_json(indent=2))\"",
        pty=True,
    )


ns = Collection()
ns.add_task(test)
ns.add_task(lint)
ns.add_task(format, name="format")
ns.add_task(check)
ns.add_task(setup)
ns.add_task(validate)
