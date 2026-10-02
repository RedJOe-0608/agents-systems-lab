from typer.testing import CliRunner

from agent_systems_lab.cli import app


def test_doctor_reports_frameworks() -> None:
    result = CliRunner().invoke(app, ["doctor"])

    assert result.exit_code == 0
    assert "langchain" in result.stdout
    assert "langgraph" in result.stdout
    assert "deepagents" in result.stdout

