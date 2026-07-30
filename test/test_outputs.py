from __future__ import annotations

import json

from eval import evaluate
from result.io import write_evaluation_outputs


def test_evaluation_writes_complete_artifacts(config, tmp_path):
    environment, metrics = evaluate(config, policy_name="heuristic")
    write_evaluation_outputs(
        tmp_path,
        config=config,
        metrics=metrics,
        schedule=environment.schedule_log,
        reconfigurations=environment.reconfiguration_log,
    )
    expected = {
        "config.json",
        "metrics.json",
        "schedule.csv",
        "reconfigurations.csv",
    }
    assert expected == {path.name for path in tmp_path.iterdir()}
    saved_metrics = json.loads(
        (tmp_path / "metrics.json").read_text(encoding="utf-8")
    )
    assert saved_metrics["completed_operations"] == 60
    assert saved_metrics["schedule_violations"] == []
