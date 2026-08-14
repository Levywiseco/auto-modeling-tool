"""Tests for run archiving, listing and comparison."""

import json
from datetime import datetime
from pathlib import Path

import pytest

from auto_modeling_tool.runs import (
    RunRecord,
    archive_run,
    compare_runs,
    format_comparison,
    format_run_table,
    list_runs,
    load_run,
    new_run_id,
    resolve_run,
)


def _make_output(tmp_path: Path, name: str = "output") -> Path:
    """A directory shaped like a finished pipeline run."""
    output = tmp_path / name
    output.mkdir()
    (output / "scoring_artifact.pkl").write_bytes(b"artifact")
    (output / "pipeline.pkl").write_bytes(b"pipeline")
    (output / "Model_Report_1.xlsx").write_bytes(b"report")
    (output / "scratch.log").write_text("ignored")
    return output


class TestRunId:
    def test_ids_are_sortable_by_time(self):
        early = new_run_id("logistic", timestamp=datetime(2026, 1, 1, 9, 0, 0))
        late = new_run_id("logistic", timestamp=datetime(2026, 1, 1, 10, 0, 0))
        assert early < late

    def test_id_carries_model_name(self):
        assert "xgboost" in new_run_id("xgboost")

    def test_same_second_runs_do_not_collide(self):
        """Ids must differ on identical inputs — the real caller varies nothing.

        This previously passed by hand-feeding different seeds, which the
        production path never does: a parameter sweep reuses one output
        directory, so every run in the same second produced one id and each
        archive silently overwrote the last.
        """
        moment = datetime(2026, 1, 1, 9, 0, 0)
        ids = {new_run_id("logistic", timestamp=moment) for _ in range(50)}
        assert len(ids) == 50

    def test_unsafe_model_names_are_sanitized(self):
        run_id = new_run_id("weird/name here")
        assert "/" not in run_id and " " not in run_id


class TestArchiveRun:
    def test_archives_artifacts_and_leaves_output_untouched(self, tmp_path):
        output = _make_output(tmp_path)
        run_dir = archive_run(
            output,
            runs_dir=tmp_path / "runs",
            config={"model_type": "logistic", "n_features": 8},
            metrics={"auc_roc": 0.71},
            selected_features=["a", "b"],
            model_type="logistic",
        )

        assert (run_dir / "scoring_artifact.pkl").read_bytes() == b"artifact"
        assert (run_dir / "pipeline.pkl").exists()
        assert (run_dir / "Model_Report_1.xlsx").exists()
        assert (run_dir / "run.json").exists()
        # Unrelated files are not swept in.
        assert not (run_dir / "scratch.log").exists()
        # The original output directory keeps working for existing tooling.
        assert (output / "scoring_artifact.pkl").exists()

    def test_records_config_metrics_and_features(self, tmp_path):
        output = _make_output(tmp_path)
        run_dir = archive_run(
            output,
            runs_dir=tmp_path / "runs",
            config={"model_type": "xgboost", "data_path": "/data/x.csv"},
            metrics={"auc_roc": 0.8, "score_psi": 0.02},
            selected_features=["f1"],
            model_type="xgboost",
        )

        payload = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        assert payload["metrics"]["auc_roc"] == 0.8
        assert payload["config"]["model_type"] == "xgboost"
        assert payload["selected_features"] == ["f1"]
        assert payload["data_path"] == "/data/x.csv"

    def test_config_snapshot_is_written(self, tmp_path):
        run_dir = archive_run(
            _make_output(tmp_path),
            runs_dir=tmp_path / "runs",
            config={"model_type": "logistic", "n_bins": 8},
            metrics={},
        )
        snapshot = run_dir / "config.yaml"
        assert snapshot.exists()
        assert "logistic" in snapshot.read_text(encoding="utf-8")

    def test_non_serializable_config_values_survive(self, tmp_path):
        run_dir = archive_run(
            _make_output(tmp_path),
            runs_dir=tmp_path / "runs",
            config={"output_dir": Path("/tmp/out"), "callback": object()},
            metrics={},
        )
        payload = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        assert payload["config"]["output_dir"] == "/tmp/out"
        assert isinstance(payload["config"]["callback"], str)

    def test_missing_output_directory_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            archive_run(tmp_path / "nope", runs_dir=tmp_path / "runs")

    def test_same_second_archives_are_all_kept(self, tmp_path):
        """A sweep whose runs share a second must keep every run."""
        moment = datetime(2026, 1, 1, 9, 0, 0)
        output = _make_output(tmp_path)
        runs_dir = tmp_path / "runs"

        for n_features in (2, 3, 4, 5, 6):
            archive_run(
                output,
                runs_dir=runs_dir,
                config={"n_features": n_features},
                metrics={"auc_roc": 0.5 + n_features / 100},
                model_type="logistic",
                timestamp=moment,
            )

        archived = list_runs(runs_dir)
        assert len(archived) == 5
        assert sorted(r.config["n_features"] for r in archived) == [2, 3, 4, 5, 6]

    def test_explicit_duplicate_run_id_is_suffixed_not_overwritten(self, tmp_path):
        output = _make_output(tmp_path)
        first = archive_run(
            output, runs_dir=tmp_path / "runs", run_id="fixed", metrics={"auc_roc": 0.1}
        )
        second = archive_run(
            output, runs_dir=tmp_path / "runs", run_id="fixed", metrics={"auc_roc": 0.9}
        )

        assert first != second
        assert load_run(first).metrics["auc_roc"] == 0.1
        assert load_run(second).metrics["auc_roc"] == 0.9

    def test_run_id_and_created_at_agree(self, tmp_path):
        """Both are derived from a single clock reading."""
        run_dir = archive_run(
            _make_output(tmp_path),
            runs_dir=tmp_path / "runs",
            metrics={},
            model_type="logistic",
        )
        record = load_run(run_dir)
        stamp = record.run_id.split("-")[0] + "-" + record.run_id.split("-")[1]
        expected = datetime.fromisoformat(record.created_at).strftime(
            "%Y%m%d-%H%M%S"
        )
        assert stamp == expected


class TestListAndResolve:
    def _archive(self, tmp_path, model, hour, tag="", **metrics):
        # A distinct output path is what separates two runs sharing a second.
        return archive_run(
            _make_output(tmp_path, f"out-{model}-{hour}{tag}"),
            runs_dir=tmp_path / "runs",
            config={"model_type": model},
            metrics=metrics,
            model_type=model,
            timestamp=datetime(2026, 1, 1, hour, 0, 0),
        )

    def test_lists_newest_first(self, tmp_path):
        self._archive(tmp_path, "logistic", 9, auc_roc=0.70)
        self._archive(tmp_path, "xgboost", 11, auc_roc=0.75)
        self._archive(tmp_path, "lightgbm", 10, auc_roc=0.72)

        records = list_runs(tmp_path / "runs")
        assert [r.model_type for r in records] == ["xgboost", "lightgbm", "logistic"]

    def test_missing_directory_is_empty_not_an_error(self, tmp_path):
        assert list_runs(tmp_path / "never-created") == []

    def test_unreadable_directories_are_skipped(self, tmp_path):
        self._archive(tmp_path, "logistic", 9, auc_roc=0.70)
        stray = tmp_path / "runs" / "not-a-run"
        stray.mkdir()
        (stray / "readme.txt").write_text("junk")
        broken = tmp_path / "runs" / "broken"
        broken.mkdir()
        (broken / "run.json").write_text("{not json")

        assert len(list_runs(tmp_path / "runs")) == 1

    def test_resolves_by_prefix_and_full_id(self, tmp_path):
        run_dir = self._archive(tmp_path, "logistic", 9, auc_roc=0.70)
        run_id = run_dir.name

        assert resolve_run(run_id, tmp_path / "runs").run_id == run_id
        assert resolve_run(run_id[:13], tmp_path / "runs").run_id == run_id
        assert resolve_run(str(run_dir)).run_id == run_id

    def test_ambiguous_prefix_raises(self, tmp_path):
        self._archive(tmp_path, "logistic", 9, tag="-a", auc_roc=0.7)
        self._archive(tmp_path, "logistic", 9, tag="-b", auc_roc=0.8)
        with pytest.raises(ValueError, match="ambiguous"):
            resolve_run("20260101-09", tmp_path / "runs")

    def test_unknown_reference_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            resolve_run("nope", tmp_path / "runs")


class TestCompare:
    def _record(self, run_id, metrics, config, features):
        return RunRecord(
            run_id=run_id,
            path=Path("/tmp") / run_id,
            metrics=metrics,
            config=config,
            selected_features=features,
        )

    def test_reports_metric_deltas(self):
        diff = compare_runs(
            self._record("a", {"auc_roc": 0.70}, {}, []),
            self._record("b", {"auc_roc": 0.75}, {}, []),
        )
        assert diff["metrics"]["auc_roc"]["delta"] == pytest.approx(0.05)

    def test_reports_only_changed_config_keys(self):
        diff = compare_runs(
            self._record("a", {}, {"model_type": "logistic", "n_bins": 10}, []),
            self._record("b", {}, {"model_type": "xgboost", "n_bins": 10}, []),
        )
        assert set(diff["config_changes"]) == {"model_type"}

    def test_ignores_paths_that_differ_every_run(self):
        diff = compare_runs(
            self._record("a", {}, {"output_dir": "/one", "runs_dir": "/r1"}, []),
            self._record("b", {}, {"output_dir": "/two", "runs_dir": "/r2"}, []),
        )
        assert diff["config_changes"] == {}

    def test_reports_feature_set_changes(self):
        diff = compare_runs(
            self._record("a", {}, {}, ["x", "y"]),
            self._record("b", {}, {}, ["y", "z"]),
        )
        assert diff["features"] == {"added": ["z"], "removed": ["x"], "shared": 1}

    def test_handles_metrics_present_on_only_one_side(self):
        diff = compare_runs(
            self._record("a", {"auc_roc": 0.7}, {}, []),
            self._record("b", {"rmse": 1.2}, {}, []),
        )
        assert diff["metrics"]["auc_roc"]["delta"] is None
        assert diff["metrics"]["rmse"]["left"] is None


class TestFormatting:
    def test_empty_history_says_so(self):
        assert "No runs" in format_run_table([])

    def test_table_lists_every_run(self, tmp_path):
        archive_run(
            _make_output(tmp_path),
            runs_dir=tmp_path / "runs",
            config={"model_type": "logistic"},
            metrics={"auc_roc": 0.7123},
            model_type="logistic",
        )
        table = format_run_table(list_runs(tmp_path / "runs"))
        assert "AUC_ROC" in table and "0.7123" in table

    def test_comparison_text_shows_direction(self):
        diff = compare_runs(
            RunRecord("a", Path("/a"), metrics={"auc_roc": 0.70}),
            RunRecord("b", Path("/b"), metrics={"auc_roc": 0.75}),
        )
        text = format_comparison(diff)
        assert "+0.0500" in text and "a  ->  b" in text


class TestRoundTrip:
    def test_archived_run_loads_back_identically(self, tmp_path):
        run_dir = archive_run(
            _make_output(tmp_path),
            runs_dir=tmp_path / "runs",
            config={"model_type": "logistic", "n_features": 8},
            metrics={"auc_roc": 0.71, "score_psi": 0.01},
            selected_features=["a", "b"],
            task="classification",
            model_type="logistic",
        )
        record = load_run(run_dir)

        assert record.run_id == run_dir.name
        assert record.task == "classification"
        assert record.metrics["auc_roc"] == 0.71
        assert record.selected_features == ["a", "b"]
        assert record.artifact_path is not None
        assert record.headline() == {"auc_roc": 0.71, "score_psi": 0.01}

    def test_load_rejects_a_non_run_directory(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_run(tmp_path)


class TestPipelineIntegration:
    """The pipeline must archive by default, and must survive a failure to."""

    def _frame(self):
        import numpy as np
        import polars as pl

        rng = np.random.default_rng(11)
        n = 240
        x1 = rng.normal(size=n)
        x2 = rng.normal(size=n)
        x3 = rng.normal(size=n)
        target = (x1 + 0.4 * x2 > 0).astype(int)
        return pl.DataFrame({
            "x1": x1, "x2": x2, "x3": x3,
            "target": target,
            "sample": ["dev"] * 180 + ["oot"] * 60,
        })

    def _run(self, tmp_path, csv, **kwargs):
        from auto_modeling_tool.main import run_modeling_pipeline

        return run_modeling_pipeline(
            str(csv),
            "target",
            output_dir=str(tmp_path / "output"),
            sample_col="sample",
            n_bins=4,
            min_samples_bin=10,
            **kwargs,
        )

    def test_run_is_archived_and_replayable(self, tmp_path):
        csv = tmp_path / "data.csv"
        self._frame().write_csv(csv)

        result = self._run(tmp_path, csv, runs_dir=str(tmp_path / "runs"))

        assert result["run_path"] is not None
        record = load_run(result["run_path"])
        assert record.metrics["auc_roc"] == pytest.approx(result["metrics"]["auc_roc"])
        assert record.selected_features == result["selected_features"]
        # The resolved config is captured, so the run can be reproduced.
        assert record.config["target_col"] == "target"
        assert record.config["n_bins"] == 4
        assert (record.path / "scoring_artifact.pkl").exists()

    def test_archiving_can_be_switched_off(self, tmp_path):
        csv = tmp_path / "data.csv"
        self._frame().write_csv(csv)

        result = self._run(
            tmp_path, csv, runs_dir=str(tmp_path / "runs"), archive_run=False
        )

        assert result["run_path"] is None
        assert not (tmp_path / "runs").exists()
        # Normal outputs are unaffected.
        assert (tmp_path / "output" / "scoring_artifact.pkl").exists()

    def test_archive_failure_does_not_lose_the_model(self, tmp_path, monkeypatch):
        csv = tmp_path / "data.csv"
        self._frame().write_csv(csv)

        import auto_modeling_tool.main as main_module

        def boom(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(main_module, "_archive_run", boom)
        result = self._run(tmp_path, csv, runs_dir=str(tmp_path / "runs"))

        assert result["run_path"] is None
        assert result["metrics"]["auc_roc"] > 0
        assert (tmp_path / "output" / "scoring_artifact.pkl").exists()
