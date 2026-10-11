"""No-training integrity tests for the accepted Q1-E7 implementation.

The only subprocess this suite may launch is the runner's read-only ``--list``.
It never calls ``--preflight`` or ``--wave``, never builds a cache, and never
invokes the FL training function.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from collections import OrderedDict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_q1_e7_cic as runner  # noqa: E402
from fl_local import e7_artifacts as artifacts  # noqa: E402
from fl_local.e7_capture import FinalModelCapture  # noqa: E402


def _tree_snapshot(root: Path) -> dict[str, tuple[int, int]]:
    if not root.exists():
        return {}
    return {
        str(p.relative_to(root)).replace("\\", "/"): (p.stat().st_size, p.stat().st_mtime_ns)
        for p in sorted(root.rglob("*")) if p.is_file()
    }


class AcceptanceAndScopeTests(unittest.TestCase):
    def test_named_acceptance_and_contract(self):
        self.assertTrue(runner.prereg_accepted())
        self.assertTrue(runner.contract_accepts_e7())

    def test_exact_job_matrix(self):
        self.assertEqual([len(runner.jobs_for_wave(w)) for w in (0, 1, 2)], [1, 7, 16])
        jobs = runner.all_jobs()
        self.assertEqual(len(jobs), 24)
        self.assertEqual(len({j.key for j in jobs}), 24)
        self.assertEqual(jobs[0].key, ("MLP", "FedAvg", "0.1", "42"))
        self.assertEqual({j.alpha for j in jobs}, {0.1})
        self.assertEqual({j.seed for j in jobs}, {42, 123, 7})

    def test_precommit_and_csv_row_keys_normalize_identically(self):
        numeric = {"Model": "MLP", "Algorithm": "FedAvg", "Alpha": 0.1, "Seed": 42}
        csv_row = {"Model": "MLP", "Algorithm": "FedAvg", "Alpha": "0.1", "Seed": "42"}
        expected = ("MLP", "FedAvg", "0.1", "42")
        self.assertEqual(runner._row_key(numeric), expected)
        self.assertEqual(runner._row_key(csv_row), expected)
        with self.assertRaises(ValueError):
            runner._row_key({**numeric, "Seed": "42.5"})

    def test_duration_fields_roundtrip_without_precision_loss(self):
        seconds = 7123.456789012345
        sec_text, min_text = runner._duration_fields(seconds)
        self.assertEqual(float(sec_text), seconds)
        self.assertEqual(float(min_text), seconds / 60.0)

    def test_staging_paths_are_unique_when_final_basenames_match(self):
        job = runner.jobs_for_wave(0)[0]
        rels = runner._artifact_relpaths(job)
        with tempfile.TemporaryDirectory() as td:
            staged = runner._staged_artifact_paths(Path(td), rels)
        self.assertEqual(len(staged), len(set(staged.values())))
        self.assertNotEqual(staged["CheckpointJSON"], staged["CellMetaJSON"])
        self.assertEqual(Path(rels["CheckpointJSON"]).name, Path(rels["CellMetaJSON"]).name)

    def test_frozen_config_and_anchors(self):
        frozen = runner.enforce_frozen_files()
        self.assertEqual(frozen["scope"]["job_keys"], [list(j.key) for j in runner.all_jobs()])
        self.assertEqual(frozen["cfg"], runner.EXPECTED_CFG)
        self.assertEqual(frozen["environment"], runner.EXPECTED_ENVIRONMENT)
        anchors = runner.load_anchors()
        self.assertEqual(set(anchors), {j.key for j in runner.all_jobs()})
        runner.verify_reference_anchors(frozen)

    def test_no_cache_build_or_unsafe_serialization_path(self):
        source = (ROOT / "run_q1_e7_cic.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        self.assertNotIn("--build-cache-only", source)
        self.assertNotIn("load_or_build_cic_cache", source)
        self.assertNotIn("torch.save", source)
        imported_modules = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        } | {
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        }
        self.assertNotIn("pickle", imported_modules)
        option_strings = {
            arg.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            for arg in node.args
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value.startswith("--")
        }
        self.assertTrue({"--list", "--preflight", "--wave", "--require-from-cache"}.issubset(option_strings))
        self.assertFalse(any(x in option_strings for x in {"--force", "--rebuild-cache", "--build-cache-only"}))

    def test_artifact_writer_imports_every_local_validation_helper(self):
        source = (ROOT / "run_q1_e7_cic.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        writer = next(
            node for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "_write_cell_artifacts"
        )
        imported = {
            alias.asname or alias.name
            for node in ast.walk(writer)
            if isinstance(node, ast.ImportFrom)
            and node.module == "fl_local.e7_artifacts"
            for alias in node.names
        }
        required = {
            "atomic_write_csv", "atomic_write_json", "atomic_write_npy",
            "atomic_write_npz", "checkpoint_payload", "compute_class_metrics",
            "matrix_rows", "sha256_file", "validate_prediction", "validate_truth",
        }
        self.assertTrue(required.issubset(imported), msg=f"missing local imports: {required-imported}")


class PureArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.truth = np.concatenate(
            [np.full(n, i, dtype=np.uint8) for i, n in enumerate(artifacts.CLASS_SUPPORTS)]
        )

    def test_perfect_and_all_zero_metrics(self):
        perfect = artifacts.compute_class_metrics(self.truth, self.truth.copy())
        self.assertEqual(perfect["accuracy"], 1.0)
        self.assertEqual(perfect["macro_f1"], 1.0)
        self.assertEqual(perfect["confusion_raw"].sum(axis=1).tolist(), list(artifacts.CLASS_SUPPORTS))
        all_zero = artifacts.compute_class_metrics(self.truth, np.zeros(len(self.truth), dtype=np.uint8))
        self.assertTrue(np.isfinite(all_zero["confusion_rownorm"]).all())
        self.assertTrue(all(np.isfinite(r["F1"]) for r in all_zero["per_class"]))
        self.assertAlmostEqual(all_zero["macro_f1"], np.mean([r["F1"] for r in all_zero["per_class"]]), places=15)

    def test_prediction_dtype_length_and_domain_fail_closed(self):
        with self.assertRaises(ValueError):
            artifacts.validate_prediction(self.truth.astype(np.int64))
        with self.assertRaises(ValueError):
            artifacts.validate_prediction(self.truth[:-1])
        bad = self.truth.copy(); bad[0] = 8
        with self.assertRaises(ValueError):
            artifacts.validate_prediction(bad)

    def test_checkpoint_roundtrip_and_corruption(self):
        state = OrderedDict([
            ("layer.weight", np.arange(12, dtype=np.float32).reshape(3, 4)),
            ("layer.bias", np.arange(3, dtype=np.float32)),
        ])
        spec, arrays = artifacts.checkpoint_payload(state)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            npz = root / "state.npz"; js = root / "state.json"
            artifacts.atomic_write_npz(npz, arrays)
            spec["npz_sha256"] = artifacts.sha256_file(npz)
            artifacts.atomic_write_json(js, spec)
            got = artifacts.read_checkpoint(npz, js)
            self.assertEqual(list(got), list(state))
            for key in state:
                self.assertTrue(np.array_equal(got[key], state[key]))
            altered = dict(spec); altered["npz_sha256"] = "0" * 64
            artifacts.atomic_write_json(js, altered)
            with self.assertRaises(ValueError):
                artifacts.read_checkpoint(npz, js)
        with self.assertRaises(ValueError):
            artifacts.checkpoint_payload({"bad": np.array([np.inf], dtype=np.float32)})
        with self.assertRaises(ValueError):
            artifacts.checkpoint_payload({"bad": np.array([object()], dtype=object)})

    def test_partition_hash_serialization(self):
        part = {0: np.array([2, 0], dtype=np.int64), 1: np.array([1], dtype=np.int64)}
        h = hashlib.sha256()
        for cid in range(2):
            idx = part[cid].astype("<i8", copy=False)
            h.update(cid.to_bytes(8, "little", signed=False))
            h.update(len(idx).to_bytes(8, "little", signed=False))
            h.update(idx.tobytes())
        self.assertEqual(artifacts.hash_partition(part, 2), h.hexdigest())
        artifacts.validate_partition(part, 3, 2)

    def test_existing_shared_truth_is_verified_before_training(self):
        frozen = {"cache": {"y_test_uint8_sha256": artifacts.sha256_bytes(self.truth.tobytes())}}
        with tempfile.TemporaryDirectory() as td:
            paths = runner.Paths(Path(td))
            truth_path = paths.out / "predictions" / "y_true_uint8.npy"
            artifacts.atomic_write_npy(truth_path, self.truth)
            doc = runner.verify_shared_truth_if_present(self.truth.astype(np.int64), frozen, paths)
            self.assertTrue(doc["exists"])
            self.assertEqual(doc["raw_sha256"], frozen["cache"]["y_test_uint8_sha256"])
            bad = self.truth.copy(); bad[0] = 1
            artifacts.atomic_write_npy(truth_path, bad)
            with self.assertRaises(RuntimeError):
                runner.verify_shared_truth_if_present(self.truth.astype(np.int64), frozen, paths)


class CaptureAndSafetyTests(unittest.TestCase):
    def setUp(self):
        import fl_local

        self.package = fl_local
        self.had_engine_attr = "engine" in fl_local.__dict__
        self.real_engine_attr = fl_local.__dict__.get("engine")
        self.real_engine = sys.modules.get("fl_local.engine")
        self.fake_engine = types.ModuleType("fl_local.engine")

        def evaluate(model, loader, device):
            return (model.value, loader, device)

        self.original = evaluate
        self.fake_engine.evaluate_model_nb = evaluate
        sys.modules["fl_local.engine"] = self.fake_engine

    def tearDown(self):
        if self.real_engine is None:
            sys.modules.pop("fl_local.engine", None)
        else:
            sys.modules["fl_local.engine"] = self.real_engine
        if self.had_engine_attr:
            self.package.engine = self.real_engine_attr
        else:
            self.package.__dict__.pop("engine", None)

    def test_capture_delegates_counts_and_restores(self):
        class FakeTensor:
            def __init__(self, value): self.value = np.asarray(value, dtype=np.float32)
            def detach(self): return self
            def cpu(self): return self
            def numpy(self): return self.value

        class Model:
            def __init__(self, value): self.value = value
            def state_dict(self): return OrderedDict([("w", FakeTensor([self.value]))])

        cap = FinalModelCapture(expected_calls=31)
        with cap:
            for i in range(31):
                result = self.fake_engine.evaluate_model_nb(Model(i), "loader", "device")
                self.assertEqual(result, (i, "loader", "device"))
            with self.assertRaises(RuntimeError):
                with FinalModelCapture():
                    pass
        self.assertIs(self.fake_engine.evaluate_model_nb, self.original)
        state = cap.export_state()
        self.assertEqual(list(state), ["w"])
        self.assertTrue(np.array_equal(state["w"], np.array([30], dtype=np.float32)))

    def test_capture_restores_on_exception_and_rejects_bad_count(self):
        cap = FinalModelCapture(expected_calls=31)
        with self.assertRaisesRegex(RuntimeError, "boom"):
            with cap:
                raise RuntimeError("boom")
        self.assertIs(self.fake_engine.evaluate_model_nb, self.original)
        with self.assertRaises(RuntimeError):
            cap.export_state()

    def test_paths_orphans_and_lock(self):
        with tempfile.TemporaryDirectory() as td:
            paths = runner.Paths(Path(td))
            with self.assertRaises(ValueError):
                runner._safe_artifact_path(paths, "../escape")
            rel = runner._artifact_relpaths(runner.jobs_for_wave(0)[0])["HistoryCSV"]
            orphan = paths.root / rel
            orphan.parent.mkdir(parents=True, exist_ok=True)
            orphan.write_text("orphan", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                runner.detect_orphans(paths)
            orphan.unlink()
            with runner.process_lock(paths):
                with self.assertRaises(RuntimeError):
                    with runner.process_lock(paths):
                        pass


class LiveReadOnlyListTest(unittest.TestCase):
    def test_list_is_the_only_live_subprocess_and_changes_nothing(self):
        e7_before = _tree_snapshot(ROOT / "q1_e7_results")
        cache_paths = [
            ROOT / "probe_results" / "cache" / "cic_27bfed227ef77fab.npz",
            ROOT / "probe_results" / "cache" / "cic_27bfed227ef77fab.meta.json",
        ]
        cache_before = [(p.stat().st_size, p.stat().st_mtime_ns) for p in cache_paths]
        proc = subprocess.run(
            [sys.executable, str(ROOT / "run_q1_e7_cic.py"), "--list"],
            cwd=ROOT, text=True, capture_output=True, timeout=120,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)
        self.assertIn("prereg accepted: True", proc.stdout)
        self.assertIn("wave 0: 0/1 done", proc.stdout)
        self.assertEqual(_tree_snapshot(ROOT / "q1_e7_results"), e7_before)
        self.assertEqual([(p.stat().st_size, p.stat().st_mtime_ns) for p in cache_paths], cache_before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
