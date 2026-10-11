"""Integrity locks for Q1-E2 (no GPU, no real N-BaIoT cache write).

Safe before and after prereg ACCEPTED: do not subprocess --wave or
--build-cache-only against the live tree (ROOT is taken from __file__,
not cwd). Mutating commands run in-process with paths redirected to a
temp dir, and run_fl_experiment is stubbed.

The AST check only flags direct subprocess.run([...]) list literals with
those flags. It does not catch Popen, os.system, assembled argv, or shell.
"""
from __future__ import annotations

import ast
import csv
import importlib
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fl_local.nbaiot_cache import load_or_build_nbaiot_cache  # noqa: E402


def _tiny_extract(base: Path) -> Path:
    extract = base / "extracted"
    cols = [f"f{i}" for i in range(12)]
    rng = np.random.RandomState(0)
    for device in ("danmini_doorbell", "ecobee_thermostat"):
        ddir = extract / device
        (ddir / "mirai_attacks").mkdir(parents=True, exist_ok=True)
        for rel, n in (("benign.csv", 40), ("mirai_attacks/ack.csv", 40)):
            df = pd.DataFrame(rng.randn(n, 12), columns=cols)
            df.to_csv(ddir / rel, index=False)
    return extract


class TestAllowBuildFalse(unittest.TestCase):
    def test_missing_cache_does_not_write_npz(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="e2_allow_build_"))
        try:
            extract = tmp / "no_such_extract"
            cache_dir = tmp / "cache"
            with self.assertRaises(FileNotFoundError):
                load_or_build_nbaiot_cache(
                    extract,
                    cache_dir,
                    max_samples_per_file=20,
                    test_fraction=0.2,
                    scale_on="train",
                    allow_build=False,
                )
            self.assertFalse(cache_dir.exists())
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_existing_extract_still_does_not_build(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="e2_allow_build_extract_"))
        try:
            extract = _tiny_extract(tmp)
            cache_dir = tmp / "cache"
            cache_dir.mkdir()
            with self.assertRaises(FileNotFoundError):
                load_or_build_nbaiot_cache(
                    extract,
                    cache_dir,
                    max_samples_per_file=20000,
                    test_fraction=0.2,
                    scale_on="train",
                    allow_build=False,
                )
            self.assertEqual(list(cache_dir.glob("*")), [])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestCidTestSchema(unittest.TestCase):
    def test_full_rebuild_omits_cid_test_train_includes_it(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="e2_cid_test_"))
        try:
            extract = _tiny_extract(tmp)
            cache_dir = tmp / "cache"
            full = load_or_build_nbaiot_cache(
                extract,
                cache_dir,
                max_samples_per_file=20000,
                test_fraction=0.2,
                scale_on="full",
            )
            train = load_or_build_nbaiot_cache(
                extract,
                cache_dir,
                max_samples_per_file=20000,
                test_fraction=0.2,
                scale_on="train",
            )
            full_npz = np.load(full["cache_path"], allow_pickle=False)
            train_npz = np.load(train["cache_path"], allow_pickle=False)
            self.assertNotIn("cid_test", full_npz.files)
            self.assertIn("cid_test", train_npz.files)
            self.assertNotEqual(full["meta"]["cache_key"], train["meta"]["cache_key"])
            self.assertEqual(full["meta"]["scale_on"], "full")
            self.assertEqual(train["meta"]["scale_on"], "train")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestProvenanceFailClosed(unittest.TestCase):
    def test_missing_json_with_complete_rows_exits_2(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="e2_prov_"))
        try:
            e2 = importlib.import_module("run_q1_e2_nbaiot")
            raw = tmp / "q1_e2_raw.csv"
            with raw.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=e2.FIELDNAMES)
                w.writeheader()
                row = {k: "" for k in e2.FIELDNAMES}
                row.update(
                    {
                        "Wave": "0",
                        "Model": "MLP",
                        "Algorithm": "FedAvg",
                        "Seed": "42",
                        "Mode": "match_notebook",
                        "ScaleOn": "train",
                        "Partition": "device",
                        "Final_F1": "0.5",
                        "N_Rounds": "30",
                    }
                )
                w.writerow(row)
            npz = tmp / "dummy.npz"
            meta = tmp / "dummy.meta.json"
            np.savez_compressed(npz, x=np.array([1], dtype=np.int64))
            meta.write_text("{}", encoding="utf-8")
            old_raw, old_prov = e2.RAW_PATH, e2.PROVENANCE
            e2.RAW_PATH = raw
            e2.PROVENANCE = tmp / "missing_provenance.json"
            try:
                with self.assertRaises(SystemExit) as ctx:
                    e2._write_provenance(
                        {
                            "cache_path": str(npz),
                            "meta": {"cache_key": "deadbeef"},
                        }
                    )
                self.assertEqual(ctx.exception.code, 2)
                self.assertFalse(e2.PROVENANCE.exists())
            finally:
                e2.RAW_PATH = old_raw
                e2.PROVENANCE = old_prov
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_orphaned_results_guard_before_cache(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="e2_orphan_"))
        try:
            e2 = importlib.import_module("run_q1_e2_nbaiot")
            raw = tmp / "q1_e2_raw.csv"
            with raw.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=e2.FIELDNAMES)
                w.writeheader()
                row = {k: "" for k in e2.FIELDNAMES}
                row.update(
                    {
                        "Final_F1": "0.5",
                        "N_Rounds": "30",
                    }
                )
                w.writerow(row)
            old_raw, old_prov = e2.RAW_PATH, e2.PROVENANCE
            e2.RAW_PATH = raw
            e2.PROVENANCE = tmp / "missing.json"
            try:
                with self.assertRaises(SystemExit) as ctx:
                    e2._fail_closed_orphaned_results()
                self.assertEqual(ctx.exception.code, 2)
            finally:
                e2.RAW_PATH = old_raw
                e2.PROVENANCE = old_prov
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


LIVE_PROVENANCE = ROOT / "q1_e2_results" / "reporting" / "Q1_E2_CACHE_PROVENANCE.json"


def _file_snapshot(path: Path) -> tuple[int, int] | None:
    if not path.is_file():
        return None
    st = path.stat()
    return (st.st_mtime_ns, st.st_size)


def _real_nbaiot_cache_snapshot() -> dict[str, tuple[int, int]]:
    cache = ROOT / "probe_results" / "cache"
    out: dict[str, tuple[int, int]] = {}
    if not cache.is_dir():
        return out
    for p in cache.glob("nbaiot_*"):
        st = p.stat()
        out[p.name] = (st.st_mtime_ns, st.st_size)
    return out


def _live_e2_artifacts() -> tuple[dict[str, tuple[int, int]], tuple[int, int] | None]:
    return _real_nbaiot_cache_snapshot(), _file_snapshot(LIVE_PROVENANCE)


def _run_main(e2, argv: list[str]) -> int:
    old = sys.argv
    sys.argv = ["run_q1_e2_nbaiot.py", *argv]
    try:
        return int(e2.main())
    finally:
        sys.argv = old


@contextmanager
def _isolate_e2(*, accepted: bool):
    e2 = importlib.import_module("run_q1_e2_nbaiot")
    tmp = Path(tempfile.mkdtemp(prefix="e2_isolate_"))
    trained = {"called": False}

    def _no_train(*_a, **_k):
        trained["called"] = True
        raise AssertionError("run_fl_experiment must not run in integrity tests")

    try:
        reporting = tmp / "q1_e2_results" / "reporting"
        reporting.mkdir(parents=True)
        prereg = reporting / "Q1_E2_PREREG_DRAFT.md"
        if accepted:
            prereg.write_text(
                "**Status:** **ACCEPTED**\n"
                "Proposer: isolation-test\n"
                "Approver: isolation-test\n"
                "Acceptance date: 2099-01-01\n",
                encoding="utf-8",
            )
        else:
            prereg.write_text(
                "**Status:** **DRAFT — not binding until dated acceptance**\n"
                "- Proposer: _[REQUIRED]_\n",
                encoding="utf-8",
            )
        before_cache, before_prov = _live_e2_artifacts()
        with (
            patch.object(e2, "ROOT", tmp),
            patch.object(e2, "WORKSPACE", tmp),
            patch.object(e2, "OUT", tmp / "q1_e2_results"),
            patch.object(e2, "CSV_DIR", tmp / "q1_e2_results" / "csv"),
            patch.object(
                e2, "RAW_PATH", tmp / "q1_e2_results" / "csv" / "q1_e2_raw.csv"
            ),
            patch.object(e2, "PREREG", prereg),
            patch.object(
                e2, "PROVENANCE", reporting / "Q1_E2_CACHE_PROVENANCE.json"
            ),
            patch.object(e2, "run_fl_experiment", _no_train),
        ):
            yield e2, tmp, trained
        after_cache, after_prov = _live_e2_artifacts()
        if after_cache != before_cache:
            raise AssertionError(
                f"live N-BaIoT cache mutated: {before_cache!r} -> {after_cache!r}"
            )
        if after_prov != before_prov:
            raise AssertionError(
                f"live E2 provenance mutated: {before_prov!r} -> {after_prov!r}"
            )
        if trained["called"]:
            raise AssertionError("run_fl_experiment was called")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


class TestRunnerGates(unittest.TestCase):
    def test_suite_source_has_no_live_mutating_subprocess(self) -> None:
        """Direct subprocess.run([...]) list literals must not pass --wave/--build-cache-only.

        Does not catch Popen, os.system, argv built at runtime, or shell strings.
        """
        src = Path(__file__).read_text(encoding="utf-8")
        tree = ast.parse(src)
        mutating = {"--wave", "--build-cache-only"}
        found: list[str] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (
                isinstance(func, ast.Attribute)
                and func.attr == "run"
                and isinstance(func.value, ast.Name)
                and func.value.id == "subprocess"
            ):
                continue
            if not node.args or not isinstance(node.args[0], ast.List):
                continue
            for elt in node.args[0].elts:
                if isinstance(elt, ast.Constant) and elt.value in mutating:
                    found.append(str(elt.value))
        self.assertEqual(found, [])

    def test_list_is_read_only_in_either_approval_state(self) -> None:
        before_cache, before_prov = _live_e2_artifacts()
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        list_p = subprocess.run(
            [sys.executable, "-u", str(ROOT / "run_q1_e2_nbaiot.py"), "--list"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            env=env,
        )
        self.assertEqual(list_p.returncode, 0, list_p.stdout + list_p.stderr)
        self.assertRegex(list_p.stdout, r"prereg accepted: (True|False)")
        after_cache, after_prov = _live_e2_artifacts()
        self.assertEqual(after_cache, before_cache)
        self.assertEqual(after_prov, before_prov)

    def test_unaccepted_wave_and_build_exit_2_isolated(self) -> None:
        before_cache, before_prov = _live_e2_artifacts()
        with _isolate_e2(accepted=False) as (e2, tmp, trained):
            self.assertEqual(
                _run_main(e2, ["--wave", "0", "--require-from-cache"]), 2
            )
            self.assertEqual(_run_main(e2, ["--build-cache-only"]), 2)
            self.assertFalse(e2._prereg_accepted())
            self.assertEqual(list((tmp / "probe_results").glob("**/*")), [])
            self.assertFalse(trained["called"])
        after_cache, after_prov = _live_e2_artifacts()
        self.assertEqual(after_cache, before_cache)
        self.assertEqual(after_prov, before_prov)

    def test_accepted_wave_and_build_do_not_touch_live_cache(self) -> None:
        before_cache, before_prov = _live_e2_artifacts()
        with _isolate_e2(accepted=True) as (e2, tmp, trained):
            self.assertTrue(e2._prereg_accepted())
            wave = _run_main(e2, ["--wave", "0", "--require-from-cache"])
            self.assertEqual(wave, 2)
            with self.assertRaises(FileNotFoundError):
                _run_main(e2, ["--build-cache-only"])
            self.assertEqual(list((tmp / "probe_results").glob("**/*")), [])
            self.assertFalse(trained["called"])
        after_cache, after_prov = _live_e2_artifacts()
        self.assertEqual(after_cache, before_cache)
        self.assertEqual(after_prov, before_prov)


if __name__ == "__main__":
    unittest.main()
