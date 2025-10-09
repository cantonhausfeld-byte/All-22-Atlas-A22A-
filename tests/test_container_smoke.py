from pathlib import Path


def test_makefile_exposes_docker_targets():
    text = Path("Makefile").read_text()
    assert "docker build -t a22a ." in text
    assert "docker run -p 8501:8501 -v $" in text


def test_schedule_workflow_exists():
    workflow = Path(".github/workflows/schedule.yml")
    assert workflow.exists(), "schedule workflow should be bootstrapped"
    content = workflow.read_text()
    assert "report_batch" in content
    assert "cron" in content
