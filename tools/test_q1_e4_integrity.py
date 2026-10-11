"""CPU-only integrity tests for the accepted Q1-E4 runner and freeze.

The only live subprocess is the literal read-only ``--list`` call.  Preflight
and wave gates are exercised in-process with temporary paths and mocks; no
training function is called and neither fixed cache can be written.
"""
from __future__ import annotations

import ast
import csv
import importlib.util
import io
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_q1_e4_train_scaler as runner  # noqa: E402
from fl_local.e4_artifacts import atomic_write_csv, atomic_write_json  # noqa: E402
from fl_local.e4_stats import (  # noqa: E402
    ALL_SEEDS,
    ARM_ORDER,
    BLOCK_ORDER,
    COMPARISON_ORDER,
    NEW_SEEDS,
    combined_estimates,
    decision,
    holm_adjust,
    paired_differences,
    paired_summary,
    prospective_analysis,
)


def _snapshot_tree(root: Path) -> dict[str, tuple[int, int]]:
    if not root.exists():
        return {}
    return {
        str(path.relative_to(root)).replace("\\", "/"): (
            int(path.stat().st_size),
            int(path.stat().st_mtime_ns),
        )
        for path in root.rglob("*")
        if path.is_file()
    }


def _cache_snapshot() -> dict[str, tuple[int, int]]:
    output = {}
    for dataset in runner.CACHE_SPECS:
        for path in (runner.LIVE.cache_npz(dataset), runner.LIVE.cache_meta(dataset)):
            output[str(path)] = (int(path.stat().st_size), int(path.stat().st_mtime_ns))
    return output


class ScopeAndAcceptanceTests(unittest.TestCase):
    def test_exact_job_matrix_and_wave_reuse(self):
        expected_counts = [4, 28, 32, 32, 16]
        jobs = runner.all_jobs()
        self.assertEqual([len(runner.jobs_for_wave(wave)) for wave in range(5)], expected_counts)
        self.assertEqual(len(jobs), 112)
        self.assertEqual(len({job.full_key for job in jobs}), 112)
        self.assertEqual({job.block for job in jobs}, {"T1", "T2", "T5", "T6"})
        self.assertEqual({job.algorithm for job in jobs}, set(runner.ALGORITHMS))
        self.assertEqual({job.seed for job in jobs}, set(runner.NEW_SEEDS))
        wave_zero = {job.key for job in runner.jobs_for_wave(0)}
        wave_one = {job.key for job in runner.jobs_for_wave(1)}
        self.assertTrue(wave_zero.isdisjoint(wave_one))
        self.assertEqual(
            wave_zero,
            {(block, "FedAvg", "127") for block in ("T1", "T2", "T5", "T6")},
        )
        self.assertTrue(all(job.seed == 127 and job.algorithm == "FedAvg" for job in runner.jobs_for_wave(0)))

    def test_named_acceptance_contract_and_exact_prereg_hash(self):
        self.assertTrue(runner.prereg_accepted())
        self.assertTrue(runner.contract_accepts_e4())
        self.assertEqual(runner._sha(runner.LIVE.prereg), runner.PREREG_SHA256)

    def test_frozen_config_and_48_anchors(self):
        frozen = runner.enforce_frozen_files()
        runner.verify_anchors(frozen)
        anchors = runner.load_anchors()
        self.assertEqual(len(anchors), 48)
        self.assertEqual(
            set(anchors),
            {
                (block, algorithm, str(seed))
                for block in ("T1", "T2", "T5", "T6")
                for algorithm in runner.ALGORITHMS
                for seed in runner.EXISTING_SEEDS
            },
        )
        self.assertEqual(frozen["scope"]["new_jobs"], 112)
        self.assertEqual(frozen["scope"]["anchor_jobs"], 48)
        self.assertEqual(frozen["scope"]["combined_jobs"], 160)
        self.assertEqual(frozen["analysis"], runner.ANALYSIS_SPEC)
        self.assertIn("tools/build_q1_e4_packet.py", frozen["file_sha256"])
        self.assertTrue(
            all(
                abs(float(row["AUC"]) - float(row["Source_AUC"])) <= 1e-15
                for row in anchors.values()
            )
        )

    def test_frozen_json_is_deterministic_and_has_no_live_timestamp(self):
        text = runner.LIVE.frozen.read_text(encoding="utf-8")
        document = json.loads(text)
        self.assertNotIn("written_utc", document)
        self.assertEqual(text, json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n")

    def test_cache_identities_exclude_legacy_full_scaler(self):
        frozen = runner.load_frozen()
        self.assertEqual(set(frozen["caches"]), {"CIC-IoT2023", "N-BaIoT"})
        for dataset, expected in runner.CACHE_SPECS.items():
            cache = frozen["caches"][dataset]
            self.assertEqual(cache["key"], expected["key"])
            self.assertEqual(cache["scale_on"], "train")
            self.assertTrue(cache["from_existing_only"])
        self.assertNotEqual(frozen["caches"]["CIC-IoT2023"]["key"], "b725017696446067")
        self.assertNotEqual(frozen["caches"]["N-BaIoT"]["key"], "a9dd0c758635aebe")

    def test_source_contains_no_cache_build_force_or_unsafe_serialization_path(self):
        paths = (
            ROOT / "run_q1_e4_train_scaler.py",
            ROOT / "tools" / "build_q1_e4_frozen_config.py",
            ROOT / "fl_local" / "e4_artifacts.py",
            ROOT / "fl_local" / "e4_stats.py",
            ROOT / "tools" / "build_q1_e4_packet.py",
        )
        forbidden_text = (
            "--build-cache-only",
            "--force",
            "load_or_build_cic_cache",
            "load_or_build_nbaiot_cache",
            "allow_pickle=True",
            "torch.save",
            "torch.load",
            "pickle.dump",
            "pickle.load",
            "os.system",
            "subprocess.Popen",
        )
        for path in paths:
            text = path.read_text(encoding="utf-8")
            ast.parse(text, filename=str(path))
            for token in forbidden_text:
                self.assertNotIn(token, text, f"forbidden token {token!r} in {path.name}")

    def test_result_schema_has_required_dataset_aware_provenance(self):
        required = {
            "Block",
            "Dataset",
            "Model",
            "Algorithm",
            "Condition",
            "Seed",
            "CacheKey",
            "NpzSha256",
            "MetaSha256",
            "PreregSha256",
            "FrozenConfigSha256",
            "PreflightSha256",
            "PartitionSha256",
            "HistorySha256",
            "CellMetaSha256",
        }
        self.assertTrue(required.issubset(runner.FIELDNAMES))
        self.assertEqual(len(runner.FIELDNAMES), len(set(runner.FIELDNAMES)))


class PureArtifactAndStatisticsTests(unittest.TestCase):
    @staticmethod
    def _complete_values(seeds):
        output = {}
        offsets = {
            "FedAvg": 0.50,
            "UnweightedFedAvg": 0.48,
            "FedTrimmedAvg": 0.53,
            "WeightedFedTrimmedAvg": 0.51,
        }
        for block_index, block in enumerate(BLOCK_ORDER):
            for seed_index, seed in enumerate(seeds):
                for arm in ARM_ORDER:
                    output[(block, arm, seed)] = offsets[arm] + block_index * 0.001 + seed_index * 0.0001
        return output

    def test_history_exact_rounds_finiteness_and_reconstruction(self):
        with tempfile.TemporaryDirectory() as temp:
            history = Path(temp) / "history.csv"
            rows = [
                {
                    "round": index,
                    "accuracy": 0.5 + index / 1000.0,
                    "f1": 0.4 + index / 1000.0,
                    "loss": 1.0 / index,
                }
                for index in range(1, 31)
            ]
            atomic_write_csv(history, ["round", "accuracy", "f1", "loss"], rows)
            values = runner._history_values(history)
            self.assertEqual(values["final_acc"], rows[-1]["accuracy"])
            self.assertEqual(values["final_f1"], rows[-1]["f1"])
            self.assertEqual(values["auc"], sum(row["f1"] for row in rows) / 30.0)
            rows[-1]["round"] = 30.5
            atomic_write_csv(history, ["round", "accuracy", "f1", "loss"], rows)
            with self.assertRaises(ValueError):
                runner._history_values(history)

    def test_unique_staging_names_when_final_basenames_match(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = runner._staged_artifact_paths(
                Path(temp),
                {
                    "HistoryCSV": "q1_e4_results/csv/same.name",
                    "CellMetaJSON": "q1_e4_results/cells/same.name",
                },
            )
            self.assertEqual(len(set(paths.values())), 2)

    def test_orphan_artifact_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = runner.Paths(Path(temp))
            orphan = paths.out / "csv" / "history_uncommitted.csv"
            orphan.parent.mkdir(parents=True)
            orphan.write_text("round,accuracy,f1,loss\n", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                runner.detect_orphans(paths)

    def test_failed_partial_commit_is_quarantined_and_resume_tree_is_clean(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = runner.Paths(Path(temp))
            job = runner.jobs_for_wave(0)[0]
            relative = runner._artifact_relpaths(job)
            final_history = paths.root / relative["HistoryCSV"]
            final_history.parent.mkdir(parents=True)
            final_history.write_text("partial", encoding="utf-8")
            stage = paths.out / ".staging" / f"{job.cell_id}.test"
            stage.mkdir(parents=True)
            (stage / "partial.tmp").write_text("partial", encoding="utf-8")
            quarantine = runner._quarantine_failed_artifacts(
                job, paths, RuntimeError("synthetic commit failure")
            )
            self.assertFalse(final_history.exists())
            self.assertFalse(stage.exists())
            self.assertTrue((quarantine / "FAILURE.json").is_file())
            runner.detect_orphans(paths)

    def test_hard_stop_recovery_is_targeted_hashed_and_resumable(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = runner.Paths(Path(temp))
            job = runner.jobs_for_wave(0)[0]
            relative = runner._artifact_relpaths(job)
            final_history = paths.root / relative["HistoryCSV"]
            final_history.parent.mkdir(parents=True)
            final_history.write_text("partial-history", encoding="utf-8")
            stage = paths.out / ".staging" / f"{job.cell_id}.test"
            stage.mkdir(parents=True)
            (stage / "partial.tmp").write_text("partial-stage", encoding="utf-8")

            entries = [
                (final_history, f"evidence__0__{final_history.name}"),
                (stage, f"evidence__1__{stage.name}"),
            ]
            quarantine = runner._write_recovery_intent(
                job, paths, entries, committed=False
            )
            first_destination = quarantine / entries[0][1]
            os.replace(final_history, first_destination)
            with self.assertRaises(RuntimeError):
                runner.detect_orphans(paths)

            with mock.patch.object(runner, "enforce_frozen_files", return_value={"ok": True}):
                resumed = runner.recover_cell(job.cell_id, paths)
            self.assertEqual(resumed, quarantine)
            self.assertFalse(stage.exists())
            self.assertTrue((quarantine / "FAILURE.json").is_file())
            intent = json.loads(
                (quarantine / "RECOVERY_INTENT.json").read_text(encoding="utf-8")
            )
            self.assertEqual(intent["status"], "complete")
            self.assertTrue(all(entry["sha256"] is not None for entry in intent["entries"]))
            runner.detect_orphans(paths)

    def test_hard_stop_recovery_refuses_other_cell_debris(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = runner.Paths(Path(temp))
            target = runner.jobs_for_wave(0)[0]
            other = runner.jobs_for_wave(0)[1]
            target_history = paths.root / runner._artifact_relpaths(target)["HistoryCSV"]
            other_history = paths.root / runner._artifact_relpaths(other)["HistoryCSV"]
            target_history.parent.mkdir(parents=True)
            target_history.write_text("target", encoding="utf-8")
            other_history.write_text("other", encoding="utf-8")
            with (
                mock.patch.object(runner, "enforce_frozen_files", return_value={"ok": True}),
                self.assertRaises(RuntimeError),
            ):
                runner.recover_cell(target.cell_id, paths)
            self.assertTrue(target_history.exists())
            self.assertTrue(other_history.exists())

    def test_hard_stop_recovery_preserves_truncated_tagged_raw_temp(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = runner.Paths(Path(temp))
            job = runner.jobs_for_wave(0)[0]
            paths.raw.parent.mkdir(parents=True)
            raw_temp = (
                paths.raw.parent
                / f".{paths.raw.name}.{job.cell_id}.synthetic.tmp"
            )
            raw_temp.write_bytes(b"Wave,Block,Data")
            with mock.patch.object(runner, "enforce_frozen_files", return_value={"ok": True}):
                quarantine = runner.recover_cell(job.cell_id, paths)
            self.assertFalse(raw_temp.exists())
            evidence = list(quarantine.glob("evidence__*"))
            self.assertEqual(len(evidence), 1)
            self.assertEqual(evidence[0].read_bytes(), b"Wave,Block,Data")
            runner.detect_orphans(paths)

    def test_atomic_raw_temp_name_is_bound_to_cell(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "raw.csv"
            moved = []

            def capture(source, destination):
                moved.append((Path(source).name, Path(destination).name))

            with mock.patch("fl_local.e4_artifacts.os.replace", side_effect=capture):
                atomic_write_csv(
                    output,
                    ["a"],
                    [{"a": 1}],
                    temp_tag="e4_T1_test",
                )
            self.assertEqual(len(moved), 1)
            self.assertTrue(moved[0][0].startswith(".raw.csv.e4_T1_test."))
            self.assertEqual(moved[0][1], "raw.csv")

    def test_duration_roundtrip_has_no_formatting_precision_loss(self):
        seconds = 7881.774860800011
        stored_seconds, stored_minutes = runner._duration_fields(seconds)
        self.assertEqual(float(stored_seconds), seconds)
        self.assertEqual(float(stored_minutes), seconds / 60.0)

    def test_paired_summary_zero_variance_rules(self):
        zeros = paired_summary([0.0] * 7, include_test=True)
        self.assertEqual(zeros["raw_p"], 1.0)
        self.assertEqual(zeros["t_statistic"], 0.0)
        self.assertEqual((zeros["ci95_low"], zeros["ci95_high"]), (0.0, 0.0))
        constant = paired_summary([0.1] * 7, include_test=True)
        self.assertEqual(constant["raw_p"], 0.0)
        self.assertIsNone(constant["t_statistic"])
        self.assertEqual((constant["ci95_low"], constant["ci95_high"]), (0.1, 0.1))

    def test_holm_uses_exact_family_and_canonical_tie_order(self):
        p_values = {key: 1.0 for key in COMPARISON_ORDER}
        p_values[COMPARISON_ORDER[0]] = 0.01
        p_values[COMPARISON_ORDER[1]] = 0.01
        adjusted = holm_adjust(p_values)
        self.assertEqual(tuple(adjusted), COMPARISON_ORDER)
        self.assertTrue(math.isclose(adjusted[COMPARISON_ORDER[0]], 0.16))
        self.assertTrue(math.isclose(adjusted[COMPARISON_ORDER[1]], 0.16))
        with self.assertRaises(ValueError):
            holm_adjust({COMPARISON_ORDER[0]: 0.01})

    def test_direction_support_reversal_and_inconclusive_are_strict(self):
        self.assertEqual(decision(0.01, 0.049, "positive"), "supported")
        self.assertEqual(decision(-0.01, 0.049, "positive"), "reversal")
        self.assertEqual(decision(0.01, 0.05, "positive"), "inconclusive")
        self.assertEqual(decision(-0.01, 0.049, "negative"), "supported")

    def test_prospective_uses_only_seven_new_seeds(self):
        new_values = self._complete_values(NEW_SEEDS)
        rows = prospective_analysis(new_values)
        self.assertEqual(len(rows), 16)
        self.assertTrue(all(row["n"] == 7 and row["df"] == 6 for row in rows))
        contaminated = dict(new_values)
        contaminated[("T1", "FedAvg", 42)] = 0.9
        with self.assertRaises(ValueError):
            prospective_analysis(contaminated)

    def test_combined_ten_seed_estimates_emit_no_decision_p_values(self):
        rows = combined_estimates(self._complete_values(ALL_SEEDS))
        self.assertEqual(len(rows), 16)
        for row in rows:
            self.assertEqual(row["n"], 10)
            self.assertEqual(row["label"], "combined data-informed estimate")
            self.assertTrue({"raw_p", "holm_p", "t_statistic", "decision"}.isdisjoint(row))

    def test_packet_schemas_keep_prospective_and_combined_rules_separate(self):
        packet_path = ROOT / "tools" / "build_q1_e4_packet.py"
        spec = importlib.util.spec_from_file_location("q1_e4_packet_test_module", packet_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        prospective = set(module.PROSPECTIVE_FIELDS)
        combined = set(module.COMBINED_FIELDS)
        self.assertTrue({"RawP", "HolmP", "Decision", "TStatistic"}.issubset(prospective))
        self.assertTrue({"RawP", "HolmP", "Decision", "TStatistic"}.isdisjoint(combined))
        self.assertEqual(len(module.PROSPECTIVE_FIELDS), len(prospective))
        self.assertEqual(len(module.COMBINED_FIELDS), len(combined))

    def test_packet_analysis_uses_exact_seed_sets_and_percentage_points(self):
        packet_path = ROOT / "tools" / "build_q1_e4_packet.py"
        spec = importlib.util.spec_from_file_location(
            "q1_e4_packet_analysis_test_module", packet_path
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        values = self._complete_values(ALL_SEEDS)
        prospective = module._prospective_rows(values)
        combined = module._combined_rows(values)
        decisions = module._block_decisions(prospective)
        interactions = module._interaction_rows(
            values, NEW_SEEDS, "synthetic descriptive interaction"
        )
        self.assertEqual(len(prospective), 16)
        self.assertEqual(len(combined), 16)
        self.assertEqual(len(interactions), 4)
        self.assertTrue(
            all(row["SeedOrder"] == "127,131,137,139,149,151,157" for row in prospective)
        )
        self.assertTrue(
            all(row["SeedOrder"] == "7,42,123,127,131,137,139,149,151,157" for row in combined)
        )
        ft_uw = next(
            row for row in prospective if row["Block"] == "T1" and row["Contrast"] == "FT-UW"
        )
        self.assertTrue(math.isclose(float(ft_uw["Mean_pp"]), 5.0))
        self.assertEqual(ft_uw["Decision"], "supported")
        self.assertEqual(decisions["T1"]["Htrim"]["decision"], "supported")
        self.assertEqual(decisions["T1"]["Hweight-hurt"]["decision"], "not_supported")
        self.assertTrue(
            all(
                {"RawP", "HolmP", "Decision", "TStatistic"}.isdisjoint(row)
                for row in combined
            )
        )

    def test_packet_seed_table_requires_all_112_new_rows(self):
        packet_path = ROOT / "tools" / "build_q1_e4_packet.py"
        spec = importlib.util.spec_from_file_location(
            "q1_e4_packet_seed_test_module", packet_path
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        metrics = {
            "Final_Acc": "0.7",
            "Final_F1": "0.6",
            "Best_Acc": "0.71",
            "Best_F1": "0.61",
            "AUC": "0.59",
            "ScaleOn": "train",
            "CacheKey": "synthetic",
            "PartitionSha256": "a" * 64,
        }
        anchors = {
            (block, algorithm, str(seed)): {
                **metrics,
                "Block": block,
                "Algorithm": algorithm,
                "Seed": str(seed),
            }
            for block in BLOCK_ORDER
            for algorithm in ARM_ORDER
            for seed in runner.EXISTING_SEEDS
        }
        raw_rows = [
            {
                **metrics,
                "Block": job.block,
                "Algorithm": job.algorithm,
                "Seed": str(job.seed),
            }
            for job in runner.all_jobs()
        ]
        rows, values = module._seed_rows(anchors, raw_rows)
        self.assertEqual((len(rows), len(values)), (160, 160))
        with self.assertRaises(RuntimeError):
            module._seed_rows(anchors, raw_rows[:-1])

    def test_incomplete_quartet_is_rejected(self):
        values = self._complete_values(NEW_SEEDS)
        del values[("T6", "WeightedFedTrimmedAvg", 157)]
        with self.assertRaises(ValueError):
            paired_differences(values, seeds=NEW_SEEDS)


class GateIsolationTests(unittest.TestCase):
    @staticmethod
    def _minimal_frozen():
        return {
            "accepted": {
                "proposer": "Sheryar Kiani",
                "approver": "Dr Hussain Kazmi",
                "date": "2026-09-26",
                "prereg_sha256": runner.PREREG_SHA256,
            },
            "baseline": {"anchor_sha256": "a" * 64},
            "caches": {
                dataset: {
                    "key": spec["key"],
                    "npz_sha256": spec["npz_sha256"],
                    "meta_sha256": spec["meta_sha256"],
                }
                for dataset, spec in runner.CACHE_SPECS.items()
            },
        }

    def test_preflight_is_no_training_and_writes_only_temp_report(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = runner.Paths(Path(temp))
            paths.frozen.parent.mkdir(parents=True)
            paths.frozen.write_text("{}\n", encoding="utf-8")
            frozen = self._minimal_frozen()
            environment = dict(runner.EXPECTED_ENVIRONMENT)
            partitions = {
                "CIC-IoT2023": {str(seed): f"cic-{seed}" for seed in runner.ALL_SEEDS},
                "N-BaIoT": {"device": "nb-device"},
            }
            fail_training = mock.Mock(side_effect=AssertionError("training must not be called"))
            with (
                mock.patch.object(runner, "prereg_accepted", return_value=True),
                mock.patch.object(runner, "contract_accepts_e4", return_value=True),
                mock.patch.object(runner, "enforce_frozen_files", return_value=frozen),
                mock.patch.object(runner, "done_keys", return_value=set()),
                mock.patch.object(runner, "detect_orphans", return_value=None),
                mock.patch.object(runner, "verify_environment", return_value=environment),
                mock.patch.object(runner, "verify_caches_and_partitions", return_value=partitions),
                mock.patch.object(runner, "verify_anchors", return_value=None),
                mock.patch("fl_local.engine.run_fl_experiment", fail_training),
            ):
                document = runner.run_preflight(paths)
            self.assertEqual(document["status"], "PASS")
            self.assertFalse(document["training_started"])
            self.assertTrue(paths.preflight.is_file())
            fail_training.assert_not_called()

    def test_wave_requires_flag_before_any_training_path(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = runner.Paths(Path(temp))
            fail_training = mock.Mock(side_effect=AssertionError("training must not be called"))
            output = io.StringIO()
            with (
                mock.patch.object(runner, "prereg_accepted", return_value=True),
                mock.patch.object(runner, "contract_accepts_e4", return_value=True),
                mock.patch("fl_local.engine.run_fl_experiment", fail_training),
                redirect_stdout(output),
            ):
                code = runner.main(["--wave", "0"], paths)
            self.assertEqual(code, 2)
            self.assertIn("requires --require-from-cache", output.getvalue())
            fail_training.assert_not_called()

    def test_recovery_cli_never_calls_training(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = runner.Paths(Path(temp))
            fail_training = mock.Mock(side_effect=AssertionError("training must not be called"))
            quarantine = paths.out / "failed_attempts" / "synthetic"
            output = io.StringIO()
            with (
                mock.patch.object(runner, "prereg_accepted", return_value=True),
                mock.patch.object(runner, "contract_accepts_e4", return_value=True),
                mock.patch.object(runner, "recover_cell", return_value=quarantine),
                mock.patch("fl_local.engine.run_fl_experiment", fail_training),
                redirect_stdout(output),
            ):
                code = runner.main(["--recover-cell", "synthetic"], paths)
            self.assertEqual(code, 0)
            self.assertIn("RECOVERY PASS", output.getvalue())
            fail_training.assert_not_called()

    def test_prior_wave_gate_fails_before_training(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = runner.Paths(Path(temp))
            frozen = self._minimal_frozen()
            fail_training = mock.Mock(side_effect=AssertionError("training must not be called"))
            with (
                mock.patch.object(runner, "prereg_accepted", return_value=True),
                mock.patch.object(runner, "contract_accepts_e4", return_value=True),
                mock.patch.object(runner, "enforce_frozen_files", return_value=frozen),
                mock.patch.object(runner, "require_preflight", return_value={}),
                mock.patch.object(runner, "verify_environment", return_value={}),
                mock.patch.object(runner, "verify_caches_and_partitions", return_value={}),
                mock.patch.object(runner, "verify_anchors", return_value=None),
                mock.patch.object(runner, "detect_orphans", return_value=None),
                mock.patch.object(runner, "_wave_complete", return_value=False),
                mock.patch("fl_local.engine.run_fl_experiment", fail_training),
            ):
                with self.assertRaises(RuntimeError):
                    runner.run_wave(1, paths)
            fail_training.assert_not_called()

    def test_process_lock_rejects_second_holder(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = runner.Paths(Path(temp))
            with runner.process_lock(paths):
                with self.assertRaises(RuntimeError):
                    with runner.process_lock(paths):
                        pass


class LiveReadOnlyListTest(unittest.TestCase):
    def test_list_is_the_only_live_subprocess_and_changes_nothing(self):
        before_tree = _snapshot_tree(runner.LIVE.out)
        before_cache = _cache_snapshot()
        completed = subprocess.run(
            [sys.executable, "-u", str(ROOT / "run_q1_e4_train_scaler.py"), "--list"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("Q1-E4 train-scaler seed expansion", completed.stdout)
        self.assertEqual(_snapshot_tree(runner.LIVE.out), before_tree)
        self.assertEqual(_cache_snapshot(), before_cache)


class TestSuiteSourceGuard(unittest.TestCase):
    def test_only_literal_list_subprocess_is_present(self):
        source = Path(__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        calls = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = ""
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                name = f"{node.func.value.id}.{node.func.attr}"
            elif isinstance(node.func, ast.Name):
                name = node.func.id
            if name in {"subprocess.run", "subprocess.Popen", "os.system", "Popen"}:
                calls.append((name, node))
        self.assertEqual(len(calls), 1)
        name, call = calls[0]
        self.assertEqual(name, "subprocess.run")
        rendered = ast.unparse(call)
        self.assertIn("'--list'", rendered)
        self.assertNotIn("--wave", rendered)
        self.assertNotIn("--preflight", rendered)


if __name__ == "__main__":
    unittest.main(verbosity=2)
