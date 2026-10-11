"""CoordMedian-Adam decision runner for the accepted 28-cell study.

A command-line start is the T1 seed-127 smoke cell only, and only after a
no-training preflight newer than the 2026-10-07T03:59:31Z report. FedNova,
matched momentum-SGD FedAvg, ``tune``, ``tune-t1-attempt2``, and the old
T1 FedAvg seed-127 smoke cell are refused.
"""
from __future__ import annotations

import gc
import io
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import fl_local.engine as engine
from fl_local.e2_partition import partition_by_device_ids
from fl_local.e4_artifacts import (
    atomic_write_bytes,
    atomic_write_json,
    partition_document,
)
from fl_local.e5_aggregators import (
    ACCEPTED_PREREG_SHA256,
    BLOCKS,
    BOUND_SHA256,
    DECISION_SEEDS,
    FEATURE_COUNTS,
    MEDIAN_FAMILY,
    decision_partition_files,
    holm_four,
    sha256_bytes,
    sha256_file,
)
from fl_local.e5_preflight import PreflightStop, verify_partition_artifact
from fl_local.e5_protocol import apply_and_read_cuda_flags, run_match_notebook_clone

HOME = "q1_e5_results/coordmedian"
# Written before this runner existed. A later no-training PASS is required.
STALE_PREFLIGHT_UTC = "2026-10-07T03:59:31Z"
CLOSED_COMMANDS = ("tune", "tune-t1-attempt2")
FORBIDDEN_ARMS = ("FedNova", "FedAvg")
OLD_SMOKE = ("T1", "FedAvg", 127)
CLASS_COUNTS = {"CIC-IoT2023": 8, "N-BaIoT": 11}
BLOCK_SPEC = {
    "T1": {"dataset": "CIC-IoT2023", "model": "MLP"},
    "T2": {"dataset": "CIC-IoT2023", "model": "CNN"},
    "T5": {"dataset": "N-BaIoT", "model": "MLP"},
    "T6": {"dataset": "N-BaIoT", "model": "CNN"},
}
CACHE_FILES = {
    "CIC-IoT2023": (
        "probe_results/cache/cic_27bfed227ef77fab.npz",
        "probe_results/cache/cic_27bfed227ef77fab.meta.json",
    ),
    "N-BaIoT": (
        "probe_results/cache/nbaiot_4ddb8b93b8821650.npz",
        "probe_results/cache/nbaiot_4ddb8b93b8821650.meta.json",
    ),
}
FROZEN_CONFIG = "q1_e4_results/reporting/Q1_E4_FROZEN_CONFIG.json"
_REPRO_READY = False


class TrainingNotAuthorized(RuntimeError):
    """Raised when a command or arm is outside the accepted CoordMedian study."""


class CoordMedianStop(RuntimeError):
    """Raised when a CoordMedian cell must pause instead of being replaced."""


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def cell_plan() -> tuple[dict, ...]:
    """Four blocks, seven decision seeds, CoordMedian-Adam at learning rate 1e-3."""
    plan = []
    for block in BLOCKS:
        spec = BLOCK_SPEC[block]
        for seed in DECISION_SEEDS:
            plan.append(
                {
                    "arm": "CoordMedian",
                    "arm_name": "CoordMedian-Adam",
                    "block": block,
                    "dataset": spec["dataset"],
                    "learning_rate": "1e-3",
                    "learning_rate_value": 1e-3,
                    "mode": "match_notebook",
                    "model": spec["model"],
                    "optimizer": "adam",
                    "seed": int(seed),
                }
            )
    if len(plan) != 28:
        raise TrainingNotAuthorized("the CoordMedian plan is not 28 cells")
    first = plan[0]
    if (first["block"], first["arm"], first["seed"]) != ("T1", "CoordMedian", 127):
        raise TrainingNotAuthorized("the first cell is not T1 CoordMedian-Adam seed 127")
    if any(cell["arm"] != "CoordMedian" or cell["learning_rate_value"] != 1e-3 for cell in plan):
        raise TrainingNotAuthorized("the plan contains a cell other than CoordMedian-Adam at 1e-3")
    return tuple(plan)


def refuse_forbidden(*, command=None, arm=None, block=None, seed=None, learning_rate=None) -> None:
    """Hard-stop closed commands, FedNova, matched FedAvg, and the old smoke cell."""
    if command in CLOSED_COMMANDS:
        raise TrainingNotAuthorized(
            "tune and tune-t1-attempt2 are closed. They will not train FedNova, "
            "matched momentum-SGD FedAvg, or the T1 FedAvg seed-127 smoke cell."
        )
    if block == OLD_SMOKE[0] and arm == OLD_SMOKE[1] and seed is not None and int(seed) == OLD_SMOKE[2]:
        raise TrainingNotAuthorized(
            "the T1 FedAvg-momentum-SGD seed-127 smoke cell is closed"
        )
    if arm in FORBIDDEN_ARMS:
        raise TrainingNotAuthorized(
            f"{arm} is closed. The CoordMedian runner will not train FedNova "
            "or matched momentum-SGD FedAvg."
        )
    if arm is not None and arm != "CoordMedian":
        raise TrainingNotAuthorized(f"arm {arm} is not CoordMedian-Adam")
    if learning_rate is not None and float(learning_rate) != 1e-3:
        raise TrainingNotAuthorized("CoordMedian learning rate is frozen at 1e-3")
    if block is not None and block not in BLOCKS:
        raise TrainingNotAuthorized(f"block {block} is outside T1, T2, T5, T6")
    if seed is not None and int(seed) not in DECISION_SEEDS:
        raise TrainingNotAuthorized(f"seed {seed} is not a decision seed")
    if command not in {
        None,
        "coordmedian",
        "coordmedian-continue",
        *CLOSED_COMMANDS,
        "wave",
        "resume",
        "preflight",
        "prepare-tuning",
    }:
        raise TrainingNotAuthorized(f"command {command} is not the CoordMedian runner")


def assert_fresh_preflight(report) -> None:
    """Accept only a later no-training PASS. The 03:59:31Z report does not open this gate."""
    if not isinstance(report, dict):
        raise TrainingNotAuthorized("preflight report is missing")
    if report.get("status") != "PASS" or report.get("kind") != "no-training-preflight":
        raise TrainingNotAuthorized("CoordMedian requires a passing no-training preflight")
    if report.get("gpu_training") is not False:
        raise TrainingNotAuthorized("preflight report records gpu training")
    if report.get("scalers_written") is not False or report.get("partition_files_written") is not False:
        raise TrainingNotAuthorized("preflight report is not a no-training record")
    if str(report.get("accepted_prereg_sha256", "")).lower() != ACCEPTED_PREREG_SHA256:
        raise TrainingNotAuthorized("preflight does not name the accepted preregistration")
    written = str(report.get("written_utc", ""))
    if written <= STALE_PREFLIGHT_UTC:
        raise TrainingNotAuthorized(
            "the no-training preflight predates the CoordMedian runner and does not authorize a cell"
        )


def load_preflight_report(root) -> dict:
    path = Path(root) / "q1_e5_results" / "Q1_E5_PREFLIGHT.json"
    if not path.is_file():
        raise TrainingNotAuthorized("no no-training preflight report is present")
    return json.loads(path.read_text(encoding="utf-8"))


def adjust_median_family(raw_p) -> dict:
    """Holm for the four median contrasts. FedNova keys are refused, not entered as tests."""
    if any(name == "FedNova-FedAvg" for _, name in raw_p):
        raise TrainingNotAuthorized("FedNova contrasts are not in this Holm family")
    if set(raw_p) != set(MEDIAN_FAMILY):
        raise TrainingNotAuthorized(
            "Holm control applies only to the four CoordMedian tests, not across E4 or a larger family"
        )
    return holm_four(raw_p)


def _cell_paths(cell) -> tuple[str, str]:
    name = f"{cell['block']}_CoordMedian_s{int(cell['seed'])}"
    return (f"{HOME}/cells/{name}.json", f"{HOME}/cells/{name}.history.json")


def _log(root: Path, message: str) -> None:
    path = Path(root) / HOME / "progress.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{_utc()} {message}\n")


def _read_json(path: Path):
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name != "nt":
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True
    # os.kill(pid, 0) raises WinError 87 on this Windows Python even when the process is alive.
    import ctypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel32.CloseHandle.restype = ctypes.c_int
    handle = kernel32.OpenProcess(0x1000, 0, int(pid))
    if not handle:
        return False
    kernel32.CloseHandle(handle)
    return True


def _acquire_pid(root: Path) -> Path:
    path = Path(root) / HOME / "coordmedian.pid"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        try:
            old = int(path.read_text(encoding="utf-8").strip())
        except ValueError:
            old = 0
        if old and old != os.getpid() and _alive(old):
            raise CoordMedianStop(f"coordmedian process {old} is already running")
    path.write_text(str(os.getpid()), encoding="utf-8")
    return path


def assess_decision_cell(row, cell) -> str:
    """Return skip, run, or stop. A paused, failed, or non-finite row is not replaced."""
    if row is None:
        return "run"
    status = str(row.get("status", ""))
    if status in {"failed", "nonfinite", "running"}:
        return "stop"
    if status != "complete":
        return "stop"
    if row.get("kind") != "decision" or row.get("decision_cell") is not True:
        return "stop"
    if row.get("arm") != "CoordMedian" or row.get("finite") is not True:
        return "stop"
    if row.get("block") != cell["block"] or int(row.get("seed", -1)) != int(cell["seed"]):
        return "stop"
    if int(row.get("rounds", 0)) != 30:
        return "stop"
    if float(row.get("learning_rate_value", -1)) != 1e-3:
        return "stop"
    try:
        final = float(row["Final_F1"])
    except (KeyError, TypeError, ValueError):
        return "stop"
    if not math.isfinite(final):
        return "stop"
    return "skip"


def _parameter_sha256(parameters) -> str:
    digest = bytearray()
    for layer in parameters:
        array = np.ascontiguousarray(np.asarray(layer, dtype="<f4"))
        digest.extend(array.tobytes(order="C"))
    return sha256_bytes(bytes(digest))


def _load_frozen(root: Path) -> dict:
    path = Path(root) / FROZEN_CONFIG
    if sha256_file(path) != BOUND_SHA256[FROZEN_CONFIG]:
        raise CoordMedianStop("frozen E4 config hash mismatch")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_sealed_cache(root: Path, dataset: str, frozen: dict) -> dict:
    npz_rel, meta_rel = CACHE_FILES[dataset]
    if sha256_file(Path(root) / npz_rel) != BOUND_SHA256[npz_rel]:
        raise CoordMedianStop(f"{dataset} cache hash mismatch")
    if sha256_file(Path(root) / meta_rel) != BOUND_SHA256[meta_rel]:
        raise CoordMedianStop(f"{dataset} cache meta hash mismatch")
    expected = frozen["caches"][dataset]
    if expected["npz_sha256"] != BOUND_SHA256[npz_rel] or expected["meta_sha256"] != BOUND_SHA256[meta_rel]:
        raise CoordMedianStop(f"{dataset} frozen cache hash does not match the bound hash")
    with np.load(Path(root) / npz_rel, allow_pickle=False) as blob:
        if sorted(blob.files) != list(expected["npz_keys"]):
            raise CoordMedianStop(f"{dataset} cache schema mismatch")
        data = {name: np.asarray(blob[name]) for name in expected["npz_keys"]}
    shape_dtype = {
        "X_train": ((expected["n_train"], expected["num_features"]), "float32"),
        "y_train": ((expected["n_train"],), "int64"),
        "X_test": ((expected["n_test"], expected["num_features"]), "float32"),
        "y_test": ((expected["n_test"],), "int64"),
        "cid_train": ((expected["n_train"],), "int64"),
    }
    if dataset == "N-BaIoT":
        shape_dtype["cid_test"] = ((expected["n_test"],), "int64")
    for name, (shape, dtype) in shape_dtype.items():
        if tuple(data[name].shape) != tuple(shape) or str(data[name].dtype) != dtype:
            raise CoordMedianStop(f"{dataset} {name} shape or dtype mismatch")
    if int(expected["num_features"]) != FEATURE_COUNTS[dataset]:
        raise CoordMedianStop(f"{dataset} feature count mismatch")
    if int(expected["num_classes"]) != CLASS_COUNTS[dataset]:
        raise CoordMedianStop(f"{dataset} class count mismatch")
    if expected.get("scale_on") != "train":
        raise CoordMedianStop(f"{dataset} cache is not scale_on=train")
    data["num_features"] = int(expected["num_features"])
    data["num_classes"] = int(expected["num_classes"])
    data["cache_sha256"] = BOUND_SHA256[npz_rel]
    return data


def _decision_partition(dataset: str, seed: int, data: dict, frozen: dict) -> tuple[dict, dict]:
    if dataset == "CIC-IoT2023":
        partition = engine.partition_dirichlet(data["y_train"], 9, 0.1, seed=int(seed))
        expected = frozen["partitions"]["CIC-IoT2023"][str(int(seed))]
        document = partition_document(partition, len(data["y_train"]), 9)
    elif dataset == "N-BaIoT":
        observed = sorted({int(value) for value in np.asarray(data["cid_test"]).tolist()})
        if observed != list(range(9)):
            raise CoordMedianStop("pooled test must contain all nine device ids")
        partition = partition_by_device_ids(np.asarray(data["cid_train"]), num_clients=9)
        expected = frozen["partitions"]["N-BaIoT"]
        document = partition_document(partition, len(data["y_train"]), 9)
        document["pooled_test_device_ids"] = list(range(9))
    else:
        raise CoordMedianStop(f"unknown dataset {dataset}")
    if document != expected:
        raise CoordMedianStop(f"{dataset} seed {seed} decision partition mismatch")
    for client_id in range(9):
        if len(partition[client_id]) == 0:
            raise CoordMedianStop(f"client {client_id} is empty")
    return partition, document


def _ensure_partition_files(root: Path, dataset: str, seed: int, partition: dict, document: dict) -> None:
    npz_rel, json_rel = decision_partition_files(
        dataset, int(seed) if dataset == "CIC-IoT2023" else None
    )
    npz_path = Path(root) / npz_rel
    json_path = Path(root) / json_rel
    if npz_path.is_file() or json_path.is_file():
        try:
            verify_partition_artifact(
                npz_path,
                json_path,
                compact=False,
                expected_sha256=document["sha256"],
            )
        except PreflightStop as exc:
            raise CoordMedianStop(f"saved decision partition mismatch: {exc}") from exc
        saved = json.loads(json_path.read_text(encoding="utf-8"))
        if saved != document:
            raise CoordMedianStop("saved decision partition does not equal the frozen entry")
        return
    stored = {
        f"c{client_id}": np.ascontiguousarray(partition[client_id], dtype="<i8")
        for client_id in range(9)
    }
    buffer = io.BytesIO()
    np.savez(buffer, **stored)
    atomic_write_bytes(npz_path, buffer.getvalue())
    atomic_write_json(json_path, document)


def _history_bytes(rows: list) -> bytes:
    return (json.dumps(rows, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _quarantine(path: Path, document: dict, status: str, reason: str) -> None:
    document["status"] = status
    document["finite"] = False
    document["finished_utc"] = _utc()
    document["reason"] = reason
    atomic_write_json(path, document)


def _enable_repro_once() -> None:
    global _REPRO_READY
    if _REPRO_READY:
        return
    engine.enable_repro_flags()
    apply_and_read_cuda_flags()
    _REPRO_READY = True


def train_coordmedian_cell(root, cell) -> dict:
    """Train one CoordMedian-Adam cell on the sealed cache. Does not call an SGD arm."""
    root = Path(root)
    refuse_forbidden(
        arm=cell.get("arm"),
        block=cell.get("block"),
        seed=cell.get("seed"),
        learning_rate=cell.get("learning_rate_value"),
    )
    if cell not in cell_plan():
        raise TrainingNotAuthorized("cell is outside the 28 CoordMedian-Adam cells")
    if not torch.cuda.is_available():
        raise CoordMedianStop("CoordMedian decision cells require CUDA")
    _enable_repro_once()
    dataset = cell["dataset"]
    seed = int(cell["seed"])
    frozen = _load_frozen(root)
    data = _load_sealed_cache(root, dataset, frozen)
    partition, partition_doc = _decision_partition(dataset, seed, data, frozen)
    _ensure_partition_files(root, dataset, seed, partition, partition_doc)
    cfg = dict(engine.CFG_FULL)
    if (
        cfg.get("lr") != 1e-3
        or cfg.get("optimizer") != "adam"
        or int(cfg.get("num_rounds", 0)) != 30
        or int(cfg.get("local_epochs", 0)) != 3
        or int(cfg.get("batch_size", 0)) != 512
        or int(cfg.get("num_clients", 0)) != 9
    ):
        raise CoordMedianStop("frozen Adam configuration is not the CoordMedian protocol")
    device = torch.device("cuda")
    engine.set_global_seed(seed)
    probe = engine.make_model(cell["model"], data["num_features"], data["num_classes"], device)
    init_hash = _parameter_sha256(engine.get_parameters_np(probe))
    del probe
    row_rel, history_rel = _cell_paths(cell)
    row_path = root / row_rel
    history_path = root / history_rel
    document = {
        "arm": "CoordMedian",
        "arm_name": "CoordMedian-Adam",
        "block": cell["block"],
        "cache_sha256": data["cache_sha256"],
        "comparator": "sealed E4 FedTrimmedAvg-Adam",
        "dataset": dataset,
        "decision_cell": True,
        "initial_parameter_sha256": init_hash,
        "kind": "decision",
        "learning_rate": "1e-3",
        "learning_rate_value": 1e-3,
        "mode": "match_notebook",
        "model": cell["model"],
        "optimizer": "adam",
        "partition_sha256": partition_doc["sha256"],
        "seed": seed,
        "started_utc": _utc(),
        "status": "running",
    }
    atomic_write_json(row_path, document)
    _log(root, f"CELL_START block={cell['block']} arm=CoordMedian seed={seed} lr=1e-3 init={init_hash}")
    stored = []

    def on_round(item: dict) -> None:
        if not all(math.isfinite(float(item[key])) for key in ("loss", "accuracy", "f1")):
            raise CoordMedianStop(f"round {item['round']} score is non-finite")
        stored.append(
            {
                "accuracy": repr(float(item["accuracy"])),
                "loss": repr(float(item["loss"])),
                "macro_f1": repr(float(item["f1"])),
                "round": int(item["round"]),
            }
        )
        payload = _history_bytes(stored)
        history_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_bytes(history_path, payload)
        _log(
            root,
            f"ROUND block={cell['block']} seed={seed} round={item['round']} "
            f"macro_f1={stored[-1]['macro_f1']} finite=true",
        )

    test_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(np.ascontiguousarray(data["X_test"])),
            torch.from_numpy(np.ascontiguousarray(data["y_test"])).long(),
        ),
        batch_size=512,
        shuffle=False,
        pin_memory=True,
    )
    try:
        result = run_match_notebook_clone(
            model_name=cell["model"],
            algorithm="CoordMedian",
            partition=partition,
            seed=seed,
            x_train=data["X_train"],
            y_train=data["y_train"],
            test_loader=test_loader,
            num_features=data["num_features"],
            num_classes=data["num_classes"],
            device=device,
            cfg=cfg,
            verbose=True,
            on_round=on_round,
        )
        if len(result["history"]) != 30 or len(stored) != 30:
            raise CoordMedianStop(f"{cell['block']} seed {seed} did not finish 30 rounds")
        payload = history_path.read_bytes()
        document.update(
            {
                "Final_Acc": stored[-1]["accuracy"],
                "Final_F1": stored[-1]["macro_f1"],
                "finished_utc": _utc(),
                "finite": True,
                "history_sha256": sha256_bytes(payload),
                "rounds": 30,
                "status": "complete",
            }
        )
        atomic_write_json(row_path, document)
        _log(
            root,
            f"CELL block={cell['block']} arm=CoordMedian seed={seed} status=complete "
            f"final_f1={document['Final_F1']}",
        )
        return document
    except CoordMedianStop as exc:
        payload = _history_bytes(stored)
        history_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_bytes(history_path, payload)
        document["history_sha256"] = sha256_bytes(payload)
        document["rounds_completed"] = len(stored)
        status = "nonfinite" if "non-finite" in str(exc) else "failed"
        _quarantine(row_path, document, status, str(exc))
        raise
    except Exception as exc:
        payload = _history_bytes(stored)
        history_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_bytes(history_path, payload)
        document["history_sha256"] = sha256_bytes(payload)
        document["rounds_completed"] = len(stored)
        reason = f"{type(exc).__name__}: {exc}"
        status = "nonfinite" if "non-finite" in reason else "failed"
        _quarantine(row_path, document, status, reason)
        raise CoordMedianStop(f"{cell['block']} CoordMedian seed {seed} failed: {reason}") from exc
    finally:
        del data, partition, test_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_coordmedian(
    root,
    *,
    report,
    limit=1,
    authorize_remaining=False,
    continuation=False,
    train_cell=None,
) -> dict:
    """Run the smoke cell, or the remaining cells after that cell is complete.

    The default call starts one cell: T1 CoordMedian-Adam seed 127. The
    continuation walks the same 28-cell plan, skips that finished cell, and
    does not call ``run_tuning``, ``run_t1_attempt2``, FedNova, or matched FedAvg.
    """
    root = Path(root)
    assert_fresh_preflight(report)
    plan = cell_plan()
    if continuation:
        if limit is not None or not authorize_remaining:
            raise TrainingNotAuthorized(
                "the remaining 27 cells start only as the authorized continuation"
            )
        smoke = plan[0]
        decision = assess_decision_cell(_read_json(root / _cell_paths(smoke)[0]), smoke)
        if decision == "stop":
            raise CoordMedianStop(
                "T1 CoordMedian seed 127 is paused, failed, or non-finite and will not be replaced"
            )
        if decision != "skip":
            raise TrainingNotAuthorized(
                "the T1 CoordMedian-Adam seed-127 smoke cell is not complete, "
                "so the other 27 cells are not started"
            )
        selected = plan
    elif limit is None:
        if not authorize_remaining:
            raise TrainingNotAuthorized(
                "Only the T1 CoordMedian-Adam seed-127 smoke cell can start. "
                "The other 27 cells are not authorized by this call."
            )
        selected = plan
    elif limit == 1:
        selected = plan[:1]
    else:
        raise TrainingNotAuthorized(
            "this runner starts the smoke cell, or the full 28-cell plan when that continuation is authorized"
        )
    train = train_coordmedian_cell if train_cell is None else train_cell
    if getattr(train, "__name__", "") in {
        "run_tuning",
        "run_t1_attempt2",
        "_train_one_cell",
        "momentum_sgd_client_update",
    }:
        raise TrainingNotAuthorized("the CoordMedian runner will not call an SGD trainer")
    pid_path = _acquire_pid(root)
    started = []
    try:
        _log(
            root,
            f"COORDMEDIAN_START planned={len(plan)} starting={len(selected)} "
            f"continuation={str(bool(continuation)).lower()} pid={os.getpid()}",
        )
        for cell in selected:
            refuse_forbidden(
                arm=cell["arm"],
                block=cell["block"],
                seed=cell["seed"],
                learning_rate=cell["learning_rate_value"],
            )
            row_path = root / _cell_paths(cell)[0]
            decision = assess_decision_cell(_read_json(row_path), cell)
            label = f"{cell['block']} CoordMedian seed {cell['seed']}"
            if decision == "stop":
                raise CoordMedianStop(f"{label} is paused, failed, or non-finite and will not be replaced")
            if decision == "skip":
                _log(root, f"CELL_SKIP block={cell['block']} arm=CoordMedian seed={cell['seed']}")
                started.append({"block": cell["block"], "seed": cell["seed"], "status": "skip"})
                continue
            train(root, cell)
            started.append({"block": cell["block"], "seed": cell["seed"], "status": "started"})
        return {
            "authorize_remaining": bool(authorize_remaining),
            "continuation": bool(continuation),
            "limit": limit,
            "planned": len(plan),
            "started_cells": started,
            "status": "complete",
        }
    except Exception as exc:
        try:
            _log(root, f"FAILED utc={_utc()} error={type(exc).__name__}: {exc}")
        except Exception:
            pass
        raise
    finally:
        if pid_path.is_file() and pid_path.read_text(encoding="utf-8").strip() == str(os.getpid()):
            pid_path.unlink(missing_ok=True)
