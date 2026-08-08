from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "paper-pilot.yml"
CANDIDATE_SHA = "0de734be8427aa3786e29062339a83b2ffb79bdd"


def test_pilot_workflow_is_pinned_and_truthfully_scoped():
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")

    assert CANDIDATE_SHA in workflow
    assert "paper-artifact-v0.5.0-rc1" in workflow
    assert "ref: ${{ env.CANDIDATE_SHA }}" in workflow
    assert "python-version: 3.11.9" in workflow
    assert "paper/requirements-linux-py311.lock" in workflow
    assert "python paper/preflight.py" in workflow
    assert "--families \"${{ matrix.family }}\"" in workflow
    assert "--command-timeout-seconds 1800" in workflow
    assert "status=verified" not in workflow
    assert "cannot be promoted" in workflow


def test_pilot_workflow_preserves_failures_and_raw_outputs():
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")

    assert "continue-on-error: true" in workflow
    assert "if: always()" in workflow
    assert "actions/upload-artifact@v4" in workflow
    assert "retention-days: 30" in workflow
    assert "Propagate matrix failure" in workflow
    assert "--resume" not in workflow
    assert "Dockerfile" not in workflow
