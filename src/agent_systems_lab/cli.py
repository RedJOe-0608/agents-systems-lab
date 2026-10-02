"""Command-line entry points for local development."""

from importlib.metadata import PackageNotFoundError, version
from platform import python_version

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(no_args_is_help=True, help="Agents Systems Lab developer utilities.")
console = Console()

_FRAMEWORKS = ("langchain", "langgraph", "langsmith", "deepagents")


def _package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "not installed"


@app.callback()
def main() -> None:
    """Inspect and operate the local agent-systems laboratory."""


@app.command()
def doctor() -> None:
    """Show the local runtime and installed framework versions."""
    table = Table(title="Agents Systems Lab")
    table.add_column("Component")
    table.add_column("Version")
    table.add_row("python", python_version())
    for package in _FRAMEWORKS:
        table.add_row(package, _package_version(package))
    console.print(table)


if __name__ == "__main__":
    app()
