"""CPU integrity tests for the accepted Q1-E5 module.

No sealed cache is written, no tuning artifact is created, and no GPU cell runs.
"""
from __future__ import annotations

import ast
import hashlib
import io
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_q1_e5 as runner  # noqa: E402
from fl_local.e4_artifacts import hash_partition  # noqa: E402
from fl_local.e5_aggregators import (  # noqa: E402
    ACCEPTED_PREREG_SHA256,
    BOUND_SHA256,
    CUDA_FLAGS,
    ENGINE_SHA256,
    FAMILY,
    FEATURE_COUNTS,
    MEDIAN_FAMILY,
    PRESENTED_PREREG_SHA256,
    SCALER_FILES,
    SCALER_FOR_BLOCK,
    client_weights,
    coefficient_sequence,
    coefficient_vector,
    coordinate_median,
    expected_local_steps,
    fedavg_server_update,
    fednova_server_update,
    index_sha256,
    normalizing_constant,
    require_finite,
    same_parameter_bytes,
    combined_index_sha256,
    lock_learning_rate,
    require_two_rows_per_stratum,
    scaler_sha256,
    holm_four,
    sha256_file,
    tuning_holdout_sizes,
    validate_compact_partition,
    vector_division_differs_from_scalar,
    weights_for_run,
)
from fl_local.e5_coordmedian import (
    CoordMedianStop,
    STALE_PREFLIGHT_UTC,
    adjust_median_family,
    assert_fresh_preflight,
    cell_plan,
    refuse_forbidden,
    run_coordmedian,
    train_coordmedian_cell,
)
from fl_local.e5_tuning import (
    TuningStop,
    _alive,
    _cell_rels,
    _run_tuning_blocks,
    assert_original_attempt_unchanged,
    assess_saved_cell,
    scan_block_cells,
    snapshot_original_attempt,
)
from fl_local.e5_preflight import (  # noqa: E402
    PREFLIGHT_CHECKS,
    PreflightStop,
    import_bound_parser,
    import_parser_file,
    read_required_csv,
    require_operation_stops,
    verify_partition_artifact,
)
from fl_local.e5_protocol import (  # noqa: E402
    FEDAVG_CLIENT_UPDATE,
    FEDNOVA_CLIENT_UPDATE,
    apply_and_read_cuda_flags,
    client_generator,
    clone_source_calls_train_client_nb,
    initial_parameter_bytes,
    make_momentum_sgd,
    minibatch_sequences,
    assert_client_endpoint,
    momentum_sgd_client_update,
    run_match_notebook_clone,
)
import fl_local.engine as engine  # noqa: E402


def _reference_scaler_hash(mean, var, scale, seen, features, subtrain_hash, validation_hash) -> str:
    digest = hashlib.sha256()
    digest.update(int(features).to_bytes(8, "little", signed=False))
    digest.update(int(seen).to_bytes(8, "little", signed=False))
    digest.update(b"<f8")
    digest.update(np.ascontiguousarray(np.asarray(mean, dtype="<f8")).tobytes(order="C"))
    digest.update(np.ascontiguousarray(np.asarray(var, dtype="<f8")).tobytes(order="C"))
    digest.update(np.ascontiguousarray(np.asarray(scale, dtype="<f8")).tobytes(order="C"))
    digest.update(bytes.fromhex(subtrain_hash))
    digest.update(bytes.fromhex(validation_hash))
    return digest.hexdigest()


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _save_partition(root: Path, npz_relative: str, partition: dict) -> None:
    npz_path = root / npz_relative
    json_path = root / (npz_relative[:-4] + ".json")
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    stored = {f"c{client_id}": np.ascontiguousarray(partition[client_id], dtype="<i8") for client_id in range(9)}
    np.savez(npz_path, **stored)
    document = {
        "sha256": hash_partition(partition),
        "client_sizes": [int(len(partition[client_id])) for client_id in range(9)],
        "client_index_sha256": [index_sha256(partition[client_id]) for client_id in range(9)],
    }
    json_path.write_text(json.dumps(document), encoding="utf-8")


def _save_scaler(root: Path, dataset: str) -> None:
    features = FEATURE_COUNTS[dataset]
    mean = np.linspace(0.0, 1.0, features, dtype="<f8")
    var = np.full(features, 1.5, dtype="<f8")
    scale = np.sqrt(var)
    subtrain = index_sha256(np.arange(features, dtype=np.int64))
    validation = index_sha256(np.arange(features, features + 3, dtype=np.int64))
    seen = np.float64(features)
    digest = scaler_sha256(
        mean_=mean,
        var_=var,
        scale_=scale,
        n_samples_seen_=seen,
        feature_count=features,
        subtrain_index_sha256=subtrain,
        validation_index_sha256=validation,
    )
    npz_relative, json_relative = SCALER_FILES[dataset]
    npz_path = root / npz_relative
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npz_path, mean_=mean, var_=var, scale_=scale)
    (root / json_relative).write_text(
        json.dumps(
            {
                "n_samples_seen_": int(seen),
                "feature_count": features,
                "dtype": "<f8",
                "subtrain_index_sha256": subtrain,
                "validation_index_sha256": validation,
                "scaler_sha256": digest,
            }
        ),
        encoding="utf-8",
    )


class FedNovaMathTests(unittest.TestCase):
    def test_median_is_the_fifth_ordered_value(self):
        clients = []
        for index in range(9):
            clients.append(
                [
                    np.array([index], dtype=np.float64),
                    np.array([[index, 8 - index]], dtype=np.float64),
                ]
            )
        median = coordinate_median(clients)
        self.assertEqual(float(median[0][0]), 4.0)
        self.assertEqual(float(median[1][0, 0]), 4.0)
        self.assertEqual(float(median[1][0, 1]), 4.0)
        with self.assertRaises(ValueError):
            coordinate_median(clients[:8])

    def test_hand_calculation_and_vector_division_fails(self):
        def hand(tau: int) -> np.ndarray:
            values = []
            previous = 0.0
            for _ in range(tau):
                previous = 0.9 * previous + 1.0
                values.append(previous)
            return np.asarray(values, dtype=np.float64)

        for tau in (1, 2, 3):
            self.assertTrue(np.array_equal(coefficient_sequence(tau), hand(tau)))
            self.assertTrue(np.array_equal(coefficient_vector(tau), hand(tau)[::-1]))
            self.assertTrue(np.isclose(normalizing_constant(tau), float(hand(tau).sum())))
        self.assertTrue(np.allclose(hand(1), [1.0]))
        self.assertTrue(np.allclose(hand(2), [1.0, 1.9]))
        self.assertTrue(np.allclose(hand(3), [1.0, 1.9, 2.71]))
        self.assertGreater(float(coefficient_vector(3)[0]), float(coefficient_vector(3)[-1]))

        taus = [1, 2, 3, 1, 2, 3, 1, 2, 3]
        areas = [normalizing_constant(tau) for tau in taus]
        start = [np.array([10.0], dtype=np.float64)]
        deltas = []
        for index, _tau in enumerate(taus):
            sign = 1.0 if index % 2 == 0 else -1.0
            deltas.append([np.array([sign * float(index + 1)], dtype=np.float64)])
        weights = np.full(9, 1.0 / 9.0)
        updated = fednova_server_update(start, deltas, weights, areas)
        tau_eff = float(np.sum(weights * np.asarray(areas, dtype=np.float64)))
        mixed = 0.0
        for index in range(9):
            normalized = float(deltas[index][0][0]) / areas[index]
            self.assertEqual(np.sign(normalized), np.sign(deltas[index][0][0]))
            mixed += float(weights[index]) * normalized
        self.assertTrue(np.allclose(updated[0], start[0] + tau_eff * mixed))
        self.assertFalse(np.allclose(fednova_server_update(start, deltas, weights, [1.0] * 9)[0], updated[0]))
        self.assertTrue(vector_division_differs_from_scalar(np.array([1.0, -2.0, 0.5]), 3))

    def test_matched_fedavg_ignores_normalizing_constant(self):
        start = [np.array([4.0, -1.0], dtype=np.float64)]
        deltas = [[np.array([float(index), -float(index)], dtype=np.float64)] for index in range(9)]
        weights = np.full(9, 1.0 / 9.0)
        areas = [float(index + 1) for index in range(9)]
        first = fedavg_server_update(start, deltas, weights, areas)
        second = fedavg_server_update(start, deltas, weights, [99.0] * 9)
        self.assertTrue(same_parameter_bytes(first, second))
        mixed = np.zeros(2, dtype=np.float64)
        for index in range(9):
            mixed = mixed + weights[index] * deltas[index][0]
        self.assertTrue(np.allclose(first[0], start[0] + mixed))

    def test_weights_use_the_rows_trained_in_that_run(self):
        subtrain = [1, 1, 1, 1, 1, 1, 1, 1, 2]
        full = [2, 2, 2, 2, 2, 2, 2, 2, 4]
        tuning = weights_for_run(subtrain, run_kind="tuning", subtrain_rows=10, full_training_rows=20)
        decision = weights_for_run(full, run_kind="decision", subtrain_rows=10, full_training_rows=20)
        self.assertTrue(np.allclose(tuning, np.asarray(subtrain, dtype=np.float64) / 10.0))
        self.assertTrue(np.allclose(decision, np.asarray(full, dtype=np.float64) / 20.0))
        self.assertTrue(np.all(tuning >= 0.0) and np.isclose(tuning.sum(), 1.0))
        self.assertTrue(np.all(decision >= 0.0) and np.isclose(decision.sum(), 1.0))
        with self.assertRaises(ValueError):
            weights_for_run(subtrain, run_kind="decision", subtrain_rows=10, full_training_rows=20)
        with self.assertRaises(ValueError):
            weights_for_run(full, run_kind="tuning", subtrain_rows=10, full_training_rows=20)
        with self.assertRaises(ValueError):
            client_weights([0, 0, 0, 0, 0, 0, 0, 0, 0], row_count=0)
        with self.assertRaises(ValueError):
            client_weights([-1, 2, 1, 1, 1, 1, 1, 1, 1], row_count=8)
        with self.assertRaises(ValueError):
            fednova_server_update(
                [np.zeros(1)],
                [[np.zeros(1)] for _ in range(9)],
                [-0.2, 0.2, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1],
                [1.0] * 9,
            )

    def test_non_finite_value_fails(self):
        with self.assertRaises(ValueError):
            require_finite(np.array([1.0, np.nan]), "fixture")
        with self.assertRaises(ValueError):
            require_finite(np.array([np.inf]), "fixture")
        with self.assertRaises(ValueError):
            fednova_server_update(
                [np.array([0.0])],
                [[np.array([np.nan])] for _ in range(9)],
                np.full(9, 1.0 / 9.0),
                [1.0] * 9,
            )


class ProtocolTests(unittest.TestCase):
    def test_counted_steps_and_second_round_resets_momentum(self):
        device = torch.device("cpu")
        model = engine.make_model("MLP", 4, 2, device)
        optimizer = make_momentum_sgd(model, 1e-2)
        self.assertEqual(optimizer.defaults["momentum"], 0.9)
        self.assertEqual(float(optimizer.defaults["dampening"]), 0.0)
        self.assertFalse(optimizer.defaults["nesterov"])
        self.assertEqual(float(optimizer.defaults["weight_decay"]), 0.0)

        start = [np.array(layer, copy=True) for layer in initial_parameter_bytes(0, 4, 2, device)]
        features = np.zeros((513, 4), dtype=np.float32)
        labels = np.zeros(513, dtype=np.int64)
        engine.set_global_seed(1)
        first = momentum_sgd_client_update(
            start,
            features,
            labels,
            lr=1e-3,
            generator=client_generator(1, 0),
            model_name="MLP",
            num_features=4,
            num_classes=2,
            device=device,
        )
        self.assertEqual(first["steps"], 3 * math.ceil(513 / 512))
        self.assertEqual(first["steps"], expected_local_steps(513))
        self.assertEqual(first["momentum_state_at_start"], 0)
        self.assertEqual(first["A_i"], normalizing_constant(first["steps"]))
        self.assertNotEqual(first["A_i"], normalizing_constant(first["steps"] * 2))

        engine.set_global_seed(1)
        second = momentum_sgd_client_update(
            first["endpoint"],
            features,
            labels,
            lr=1e-3,
            generator=client_generator(1, 0),
            model_name="MLP",
            num_features=4,
            num_classes=2,
            device=device,
        )
        self.assertEqual(second["momentum_state_at_start"], 0)
        self.assertEqual(second["steps"], first["steps"])
        self.assertEqual(second["A_i"], normalizing_constant(second["steps"]))

    def test_same_client_update_is_byte_identical_before_aggregation(self):
        self.assertIs(FEDNOVA_CLIENT_UPDATE, FEDAVG_CLIENT_UPDATE)
        self.assertIs(FEDNOVA_CLIENT_UPDATE, momentum_sgd_client_update)
        device = torch.device("cpu")
        start = [np.array(layer, copy=True) for layer in initial_parameter_bytes(3, 4, 2, device)]
        features = np.arange(40, dtype=np.float32).reshape(10, 4)
        labels = np.arange(10, dtype=np.int64) % 2

        def run_once(parameters):
            engine.set_global_seed(11)
            return momentum_sgd_client_update(
                parameters,
                features,
                labels,
                lr=1e-2,
                generator=client_generator(11, 2),
                model_name="MLP",
                num_features=4,
                num_classes=2,
                device=device,
                local_epochs=1,
                batch_size=4,
            )

        first = run_once(start)
        second = run_once(start)
        self.assertTrue(same_parameter_bytes(first["endpoint"], second["endpoint"]))
        self.assertTrue(same_parameter_bytes(first["delta"], second["delta"]))
        shifted = [np.array(layer, copy=True) for layer in start]
        shifted[0] = shifted[0] + 1.0
        third = run_once(shifted)
        self.assertFalse(same_parameter_bytes(first["delta"], third["delta"]))

    def test_shared_seed_repeats_init_partition_order_and_minibatches(self):
        device = torch.device("cpu")
        tuning = [initial_parameter_bytes(10007, 4, 3, device) for _ in range(6)]
        for item in tuning[1:]:
            self.assertTrue(same_parameter_bytes(tuning[0], item))
        decision = [initial_parameter_bytes(127, 4, 3, device) for _ in range(2)]
        self.assertTrue(same_parameter_bytes(decision[0], decision[1]))

        labels = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0], dtype=np.int64)
        first = engine.partition_dirichlet(labels, 9, 0.1, 10007)
        second = engine.partition_dirichlet(labels, 9, 0.1, 10007)
        self.assertEqual(list(first), list(range(9)))
        for client_id in range(9):
            self.assertTrue(np.array_equal(first[client_id], second[client_id]))

        sequences = [
            minibatch_sequences(20, seed=10007, client_index=3, rounds=2, batch_size=5, persistent=True)
            for _ in range(6)
        ]
        for sequence in sequences[1:]:
            self.assertEqual(sequence, sequences[0])
        left = minibatch_sequences(20, seed=127, client_index=3, rounds=2, batch_size=5, persistent=True)
        right = minibatch_sequences(20, seed=127, client_index=3, rounds=2, batch_size=5, persistent=True)
        self.assertEqual(left, right)

    def test_persistent_generator_advances_and_recreation_repeats(self):
        persistent = minibatch_sequences(
            30, seed=19, client_index=4, rounds=2, batch_size=10, persistent=True
        )
        recreated = minibatch_sequences(
            30, seed=19, client_index=4, rounds=2, batch_size=10, persistent=False
        )
        self.assertNotEqual(persistent[0], persistent[1])
        self.assertEqual(recreated[0], recreated[1])
        self.assertEqual(persistent[0], recreated[0])

    def test_cuda_flags_are_set_and_read_back(self):
        self.assertEqual(apply_and_read_cuda_flags(), CUDA_FLAGS)

    def test_clone_matches_match_notebook_and_passes_no_generator(self):
        self.assertTrue(clone_source_calls_train_client_nb())
        rng = np.random.default_rng(0)
        features = rng.normal(size=(45, 4)).astype(np.float32)
        labels = np.arange(45) % 3
        partition = {client_id: np.arange(client_id, 45, 9) for client_id in range(9)}
        test_x = rng.normal(size=(12, 4)).astype(np.float32)
        test_y = np.arange(12) % 3
        loader = DataLoader(
            TensorDataset(torch.tensor(test_x), torch.tensor(test_y, dtype=torch.long)),
            batch_size=4,
        )
        cfg = {
            "num_clients": 9,
            "num_rounds": 1,
            "local_epochs": 1,
            "batch_size": 8,
            "lr": 1e-3,
            "trimmed_fraction": 0.1,
            "optimizer": "adam",
            "sgd_momentum": 0.9,
        }
        device = torch.device("cpu")
        calls = []
        real_train = engine.train_client_nb

        def spy_train(*args, **kwargs):
            calls.append(kwargs)
            return real_train(*args, **kwargs)

        with mock.patch.object(engine, "train_client_nb", spy_train):
            clone = run_match_notebook_clone(
                model_name="MLP",
                algorithm="FedTrimmedAvg",
                partition=partition,
                seed=5,
                x_train=features,
                y_train=labels,
                test_loader=loader,
                num_features=4,
                num_classes=3,
                device=device,
                cfg=cfg,
                verbose=False,
            )
        self.assertEqual(len(calls), 9)
        self.assertTrue(all("generator" not in kwargs for kwargs in calls))

        captured = []
        real_set = engine.set_parameters_np

        def spy_set(model, parameters):
            captured.append([np.array(layer, copy=True) for layer in parameters])
            return real_set(model, parameters)

        with mock.patch.object(engine, "set_parameters_np", spy_set):
            result = engine.run_fl_experiment(
                model_name="MLP",
                algorithm="FedTrimmedAvg",
                partition=partition,
                alpha_label="0.1",
                seed=5,
                X_train=features,
                y_train=labels,
                test_loader=loader,
                num_features=4,
                num_classes=3,
                device=device,
                cfg=cfg,
                mode="match_notebook",
                verbose=False,
            )
        self.assertTrue(captured)
        self.assertTrue(same_parameter_bytes(clone["parameters"], captured[-1]))
        self.assertEqual(clone["history"], result["history"])


class ArtifactTests(unittest.TestCase):
    def test_compact_indices_reject_a_full_training_position(self):
        partition = {client_id: np.array([client_id], dtype=np.int64) for client_id in range(9)}
        validate_compact_partition(partition, 9)
        partition[3] = np.array([4825692], dtype=np.int64)
        with self.assertRaises(ValueError):
            validate_compact_partition(partition, 9)

    def test_scaler_hash_covers_the_required_fields_and_blocks_share_artifacts(self):
        self.assertEqual(SCALER_FILES[SCALER_FOR_BLOCK["T1"]], SCALER_FILES[SCALER_FOR_BLOCK["T2"]])
        self.assertEqual(SCALER_FILES[SCALER_FOR_BLOCK["T5"]], SCALER_FILES[SCALER_FOR_BLOCK["T6"]])
        self.assertNotEqual(SCALER_FOR_BLOCK["T1"], SCALER_FOR_BLOCK["T5"])
        self.assertEqual(FEATURE_COUNTS["CIC-IoT2023"], 39)
        self.assertEqual(FEATURE_COUNTS["N-BaIoT"], 115)
        mean = np.array([1.0, 2.0], dtype="<f8")
        var = np.array([3.0, 4.0], dtype="<f8")
        scale = np.array([5.0, 6.0], dtype="<f8")
        subtrain = index_sha256(np.array([0, 1, 2], dtype=np.int64))
        validation = index_sha256(np.array([3, 4], dtype=np.int64))
        digest = scaler_sha256(
            mean_=mean,
            var_=var,
            scale_=scale,
            n_samples_seen_=np.float64(10.0),
            feature_count=2,
            subtrain_index_sha256=subtrain,
            validation_index_sha256=validation,
        )
        self.assertEqual(
            digest,
            _reference_scaler_hash(mean, var, scale, 10, 2, subtrain, validation),
        )
        base = {
            "mean_": mean,
            "var_": var,
            "scale_": scale,
            "n_samples_seen_": 10,
            "feature_count": 2,
            "subtrain_index_sha256": subtrain,
            "validation_index_sha256": validation,
        }

        def altered(**updates):
            payload = dict(base)
            payload.update(updates)
            return scaler_sha256(**payload)

        self.assertNotEqual(digest, altered(mean_=mean + 1))
        self.assertNotEqual(digest, altered(var_=var + 1))
        self.assertNotEqual(digest, altered(scale_=scale + 1))
        self.assertNotEqual(digest, altered(n_samples_seen_=11))
        self.assertNotEqual(digest, altered(subtrain_index_sha256=validation, validation_index_sha256=subtrain))
        with self.assertRaises(ValueError):
            altered(n_samples_seen_=np.float64(10.5))

    def test_array_bytes_accept_and_compressed_npz_hash_does_not(self):
        payload = np.arange(6, dtype=np.float32).reshape(2, 3)
        copy = np.array(payload, copy=True)
        self.assertTrue(np.shares_memory(payload, payload))
        self.assertTrue(
            hashlib.sha256(np.ascontiguousarray(payload).tobytes(order="C")).hexdigest()
            == hashlib.sha256(np.ascontiguousarray(copy).tobytes(order="C")).hexdigest()
        )
        from fl_local.e5_aggregators import arrays_byte_equal

        self.assertTrue(arrays_byte_equal(payload, copy))
        changed = copy.copy()
        changed[0, 0] = 99
        self.assertFalse(arrays_byte_equal(payload, changed))
        first = io.BytesIO()
        second = io.BytesIO()
        np.savez_compressed(first, X=payload)
        np.savez_compressed(second, X=payload, extra=np.array([1], dtype=np.int64))
        first_hash = hashlib.sha256(first.getvalue()).hexdigest()
        second_hash = hashlib.sha256(second.getvalue()).hexdigest()
        self.assertNotEqual(first_hash, second_hash)
        reloaded_first = np.load(io.BytesIO(first.getvalue()))["X"]
        reloaded_second = np.load(io.BytesIO(second.getvalue()))["X"]
        self.assertTrue(arrays_byte_equal(reloaded_first, reloaded_second))
        self.assertNotEqual(first_hash, second_hash)

    def test_csv_and_parser_failures_stop_and_skip_branch_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            missing = root / "missing.csv"
            with self.assertRaises(PreflightStop):
                read_required_csv(missing, min_columns=5)
            empty = root / "empty.csv"
            empty.write_text("a,b,c,d,e\n", encoding="utf-8")
            with self.assertRaises(PreflightStop):
                read_required_csv(empty, min_columns=5)
            short = root / "short.csv"
            short.write_text("a,b\n1,2\n", encoding="utf-8")
            with self.assertRaises(PreflightStop):
                read_required_csv(short, min_columns=5)
            parser = root / "bad_parser.py"
            parser.write_text("raise RuntimeError('import boom')\n", encoding="utf-8")
            with self.assertRaises(PreflightStop):
                import_parser_file(parser)

            def skip_csv():
                try:
                    read_required_csv(missing, min_columns=5)
                except Exception:
                    return None

            def skip_parser():
                try:
                    import_parser_file(parser)
                except Exception:
                    return None

            with self.assertRaises(AssertionError):
                require_operation_stops(skip_csv)
            with self.assertRaises(AssertionError):
                require_operation_stops(skip_parser)
            require_operation_stops(lambda: read_required_csv(missing, min_columns=5))
            require_operation_stops(lambda: import_parser_file(parser))
        with self.assertRaises(PreflightStop):
            import_bound_parser("fl_local.not_a_real_e5_parser")
        self.assertIsNotNone(import_bound_parser("fl_local.cic_cache"))
        self.assertIsNotNone(import_bound_parser("fl_local.nbaiot_cache"))


class GateTests(unittest.TestCase):
    def test_engine_and_accepted_preregistration_hashes(self):
        self.assertEqual(sha256_file(ROOT / "fl_local" / "engine.py"), ENGINE_SHA256)
        prereg = sha256_file(ROOT / "q1_e5_results" / "Q1_E5_PREREG_DRAFT.md")
        self.assertEqual(prereg, ACCEPTED_PREREG_SHA256)
        self.assertNotEqual(prereg, PRESENTED_PREREG_SHA256)
        for relative in (
            "fl_local/cic_cache.py",
            "fl_local/nbaiot_cache.py",
            "fl_local/e4_artifacts.py",
            "fl_local/e2_partition.py",
            "probe_results/cache/cic_27bfed227ef77fab.meta.json",
            "probe_results/cache/nbaiot_4ddb8b93b8821650.meta.json",
            "q1_e4_results/reporting/Q1_E4_FROZEN_CONFIG.json",
            "q1_e4_results/reporting/q1_e4_seed.csv",
        ):
            self.assertEqual(sha256_file(ROOT / relative), BOUND_SHA256[relative])

    def test_preflight_does_not_require_scaler_files(self):
        self.assertFalse(any("scaler" in item for item in PREFLIGHT_CHECKS))
        self.assertEqual(len(runner.required_scaler_rels()), 4)

    def test_entry_without_recheck_fails_including_coordmedian_only(self):
        called = []

        def action():
            called.append("trained")

        for command in ("tune", "tune-t1-attempt2"):
            with self.assertRaises(runner.TrainingNotAuthorized) as caught:
                runner.invoke(
                    command,
                    recheck=runner.RecheckReceipt(command),
                    action=action,
                    allow_gpu=True,
                )
            self.assertIn("closed", str(caught.exception))
        self.assertEqual(called, [])
        for command in ("wave", "resume"):
            with self.assertRaises(RuntimeError) as caught:
                runner.invoke(
                    command,
                    recheck=None,
                    action=action,
                    allow_gpu=True,
                    coordmedian_only=(command == "resume"),
                )
            self.assertIn("Section 9.1", str(caught.exception))
        self.assertEqual(called, [])
        tune_receipt = runner.RecheckReceipt("tune")
        with self.assertRaises(RuntimeError):
            runner.invoke(
                "resume",
                recheck=tune_receipt,
                action=action,
                allow_gpu=True,
                coordmedian_only=True,
            )
        receipt = runner.RecheckReceipt("resume", coordmedian_only=True)
        with self.assertRaises(runner.TrainingNotAuthorized):
            runner.invoke("resume", recheck=receipt, action=action, allow_gpu=False, coordmedian_only=True)
        self.assertEqual(called, [])
        self.assertEqual(
            runner.invoke("resume", recheck=receipt, action=action, allow_gpu=True, coordmedian_only=True),
            None,
        )
        self.assertEqual(called, ["trained"])

    def test_cli_rechecks_before_training_and_keeps_waves_closed(self):
        source = Path(runner.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        names = []
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == "main":
                for child in ast.walk(node):
                    if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                        names.append(child.func.id)
        self.assertLess(names.index("section_91_recheck"), names.index("invoke"))
        for command in ("tune", "tune-t1-attempt2"):
            with mock.patch("run_q1_e5.section_91_recheck", side_effect=AssertionError("recheck")), mock.patch(
                "run_q1_e5.run_tuning", side_effect=AssertionError("tuned")
            ), mock.patch("run_q1_e5.run_t1_attempt2", side_effect=AssertionError("attempt2")):
                with self.assertRaises(runner.TrainingNotAuthorized) as caught:
                    runner.main([command])
            self.assertIn("closed", str(caught.exception))
        for command in ("wave", "resume"):
            argv = [command, "--coordmedian-only"] if command == "resume" else [command]
            with mock.patch("run_q1_e5.section_91_recheck", side_effect=RuntimeError("recheck-called")):
                with self.assertRaises(RuntimeError) as caught:
                    runner.main(argv)
            self.assertEqual(str(caught.exception), "recheck-called")
        for command in ("wave", "resume"):
            argv = [command, "--coordmedian-only"] if command == "resume" else [command]
            receipt = runner.RecheckReceipt(command, coordmedian_only=(command == "resume"))
            with mock.patch("run_q1_e5.section_91_recheck", return_value=receipt), mock.patch(
                "run_q1_e5.run_tuning"
            ) as tuned:
                with self.assertRaises(runner.TrainingNotAuthorized):
                    runner.main(argv)
            tuned.assert_not_called()
        with mock.patch("run_q1_e5.run_preflight", return_value={"status": "PASS"}) as preflight, mock.patch(
            "run_q1_e5.section_91_recheck"
        ) as recheck, mock.patch("run_q1_e5.run_tuning") as tuned:
            self.assertEqual(runner.main(["preflight"]), 0)
        preflight.assert_called_once()
        recheck.assert_not_called()
        tuned.assert_not_called()
        with mock.patch(
            "run_q1_e5.prepare_tuning_artifacts", return_value={"manifest_path": "ok"}
        ) as prepare, mock.patch("run_q1_e5.section_91_recheck") as recheck, mock.patch(
            "run_q1_e5.run_tuning"
        ) as tuned:
            self.assertEqual(runner.main(["prepare-tuning"]), 0)
        prepare.assert_called_once()
        recheck.assert_not_called()
        tuned.assert_not_called()
        with self.assertRaises(runner.TrainingNotAuthorized) as smoke:
            runner.main(["wave", "--wave", "0"])
        self.assertIn("smoke cell", str(smoke.exception))


class TuningSelectionTests(unittest.TestCase):
    def test_holdout_sizes_follow_the_sklearn_ceil_rule(self):
        self.assertEqual(tuning_holdout_sizes(4_825_692), (3_860_553, 965_139))
        self.assertEqual(tuning_holdout_sizes(1_418_112), (1_134_489, 283_623))
        self.assertEqual(math.ceil(0.2 * 4_825_692), 965_139)
        self.assertEqual(math.ceil(0.2 * 1_418_112), 283_623)

    def test_a_stratum_with_one_row_stops(self):
        require_two_rows_per_stratum(np.array([0, 0, 1, 1]))
        with self.assertRaises(ValueError):
            require_two_rows_per_stratum(np.array([0, 0, 1]))

    def test_combined_index_hash_is_order_sensitive(self):
        subtrain = np.array([0, 1, 2], dtype=np.int64)
        validation = np.array([3, 4], dtype=np.int64)
        forward = combined_index_sha256(subtrain, validation)
        backward = combined_index_sha256(validation, subtrain)
        self.assertNotEqual(forward, backward)
        self.assertEqual(forward, combined_index_sha256(subtrain, validation.copy()))

    def test_lock_prefers_the_higher_mean_and_the_smaller_rate_on_a_tie(self):
        higher = {
            "0.001": {"FedAvg": 0.1, "FedNova": 0.1},
            "0.01": {"FedAvg": 0.2, "FedNova": 0.2},
            "0.1": {"FedAvg": 0.8, "FedNova": 0.4},
        }
        self.assertEqual(lock_learning_rate(higher), "0.1")
        tie_at_small_end = {
            "0.001": {"FedAvg": 0.5, "FedNova": 0.5},
            "0.01": {"FedAvg": 0.25, "FedNova": 0.75},
            "0.1": {"FedAvg": 0.0, "FedNova": 0.0},
        }
        self.assertEqual(lock_learning_rate(tie_at_small_end), "0.001")
        tie_between_larger_rates = {
            "0.001": {"FedAvg": 0.1, "FedNova": 0.1},
            "0.01": {"FedAvg": 0.5, "FedNova": 0.5},
            "0.1": {"FedAvg": 0.25, "FedNova": 0.75},
        }
        self.assertEqual(lock_learning_rate(tie_between_larger_rates), "0.01")
        with self.assertRaises(ValueError):
            lock_learning_rate({"0.001": {"FedAvg": 0.5, "FedNova": 0.5}})
        incomplete = {
            "0.001": {"FedAvg": float("nan"), "FedNova": 0.5},
            "0.01": {"FedAvg": 0.5, "FedNova": 0.5},
            "0.1": {"FedAvg": 0.5, "FedNova": 0.5},
        }
        with self.assertRaises(ValueError):
            lock_learning_rate(incomplete)

    def test_a_failed_or_incomplete_cell_does_not_count_as_skippable(self):
        self.assertEqual(assess_saved_cell(None), "run")
        self.assertEqual(assess_saved_cell({"status": "running"}), "run")
        self.assertEqual(assess_saved_cell({"status": "failed", "finite": False}), "stop")
        self.assertEqual(assess_saved_cell({"status": "nonfinite", "finite": False}), "stop")
        complete = {
            "kind": "tuning",
            "status": "complete",
            "decision_cell": False,
            "finite": True,
            "steps_checked": True,
            "seed": 10007,
            "rounds": 30,
            "Final_F1": "0.5",
        }
        self.assertEqual(assess_saved_cell(complete), "skip")
        missing_score = dict(complete)
        missing_score["Final_F1"] = "nan"
        self.assertEqual(assess_saved_cell(missing_score), "stop")

    def test_recheck_stops_for_a_missing_scaler_even_for_coordmedian(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected = {}
            for relative in BOUND_SHA256:
                payload = f"bound:{relative}".encode()
                _write_bytes(root / relative, payload)
                expected[relative] = hashlib.sha256(payload).hexdigest()
            with self.assertRaises(PreflightStop) as caught:
                runner.section_91_recheck(
                    root,
                    command="resume",
                    coordmedian_only=True,
                    expected_hashes=expected,
                    check_environment=False,
                    apply_cuda=False,
                )
            self.assertIn("tuning scaler missing", str(caught.exception))

            for dataset in ("CIC-IoT2023", "N-BaIoT"):
                _save_scaler(root, dataset)
            partition = {
                client_id: np.arange(client_id, 18, 9, dtype=np.int64) for client_id in range(9)
            }
            for block in ("T1", "T2", "T5", "T6"):
                _save_partition(root, runner.tuning_partition_files(block)[0], partition)
            receipt = runner.section_91_recheck(
                root,
                command="tune",
                expected_hashes=expected,
                check_environment=False,
                apply_cuda=False,
            )
            self.assertEqual(receipt.command, "tune")
            self.assertEqual(receipt.token, "section-9.1")

            broken = dict(partition)
            broken[0] = np.array([0, 4825692], dtype=np.int64)
            _save_partition(root, runner.tuning_partition_files("T1")[0], broken)
            with self.assertRaises(PreflightStop):
                verify_partition_artifact(
                    root / runner.tuning_partition_files("T1")[0],
                    root / runner.tuning_partition_files("T1")[1],
                    compact=True,
                )
            payload = (root / "fl_local" / "engine.py").read_bytes() + b"x"
            (root / "fl_local" / "engine.py").write_bytes(payload)
            with self.assertRaises(PreflightStop) as mismatch:
                runner.section_91_recheck(
                    root,
                    command="tune",
                    expected_hashes=expected,
                    check_environment=False,
                    apply_cuda=False,
                )
            self.assertIn("bound hash mismatch", str(mismatch.exception))


class EndpointAndReloadTests(unittest.TestCase):
    def test_large_finite_float32_endpoints_reconstruct_exactly(self):
        rng = np.random.default_rng(0)
        start = rng.standard_normal(4096).astype(np.float32) * np.float32(1e20)
        endpoint = rng.standard_normal(4096).astype(np.float32) * np.float32(1e20)
        delta = endpoint.astype(np.float64) - start.astype(np.float64)
        float32_round_trip = start + (endpoint - start)
        self.assertFalse(np.array_equal(float32_round_trip, endpoint))
        assert_client_endpoint(start, delta, endpoint)

        huge = np.array([np.float32(np.finfo(np.float32).max / 2)], dtype=np.float32)
        neighbor = np.nextafter(huge, np.float32(np.finfo(np.float32).max))
        assert_client_endpoint(
            huge,
            neighbor.astype(np.float64) - huge.astype(np.float64),
            neighbor,
        )

    def test_non_finite_endpoint_is_not_reported_as_a_reconstruction_mismatch(self):
        start = np.array([np.float32(1e20)], dtype=np.float32)
        for value in (np.nan, np.inf):
            endpoint = np.array([np.float32(value)], dtype=np.float32)
            delta = endpoint.astype(np.float64) - start.astype(np.float64)
            with self.assertRaises(RuntimeError) as caught:
                assert_client_endpoint(start, delta, endpoint)
            message = str(caught.exception)
            self.assertIn("non-finite", message)
            self.assertNotIn("not the start plus delta_i", message)

    def _bindings(self, stamp: str) -> dict:
        return {
            "combined_index_sha256": stamp,
            "dataset": "CIC-IoT2023",
            "matrix_sha256": "matrix",
            "n_subtrain": 8,
            "partition_sha256": "partition",
            "scaler_sha256": "scaler",
        }

    def test_each_tuning_cell_reloads_its_own_artifacts(self):
        import fl_local.e5_tuning as tuning

        partitions = []

        def fake_reload(root, block):
            del root, block
            stamp = f"{len(partitions)}"
            partition = {client_id: np.array([len(partitions)], dtype=np.int64) for client_id in range(9)}
            partitions.append(partition)
            return self._bindings("index"), partition

        def fake_disposition(root, block, arm, rate_token, bindings, **_extra):
            del root, block, arm, rate_token, bindings, _extra
            return "skip"

        with mock.patch.object(tuning, "reload_tuning_cell", fake_reload), mock.patch.object(
            tuning, "cell_disposition", fake_disposition
        ):
            plan = scan_block_cells(Path("."), "T1")
        self.assertEqual(len(plan), 6)
        self.assertEqual(len(partitions), 6)
        self.assertEqual(len({id(item) for item in partitions}), 6)
        self.assertEqual([int(item[4][0][0]) for item in plan], list(range(6)))

    def test_a_failed_cell_stops_during_its_own_reload_and_is_not_replaced(self):
        import fl_local.e5_tuning as tuning

        calls = []

        def fake_reload(root, block):
            del root, block
            calls.append(len(calls))
            return self._bindings("index"), {0: np.array([0], dtype=np.int64)}

        def fake_disposition(root, block, arm, rate_token, bindings, **_extra):
            del root, bindings, _extra
            if (arm, rate_token) == ("FedNova", "0.1"):
                return "stop"
            return "skip"

        with mock.patch.object(tuning, "reload_tuning_cell", fake_reload), mock.patch.object(
            tuning, "cell_disposition", fake_disposition
        ):
            with self.assertRaises(TuningStop) as caught:
                scan_block_cells(Path("."), "T1")
        self.assertEqual(len(calls), 6)
        self.assertIn("T1 FedNova lr=0.1 is failed or non-finite and will not be replaced", str(caught.exception))

    def test_a_hash_change_between_cells_stops_the_block(self):
        import fl_local.e5_tuning as tuning

        def fake_reload(root, block):
            del root, block
            stamp = "first" if not hasattr(fake_reload, "used") else "second"
            fake_reload.used = True
            return self._bindings(stamp), {0: np.array([0], dtype=np.int64)}

        def fake_disposition(root, block, arm, rate_token, bindings, **_extra):
            del root, block, arm, rate_token, bindings, _extra
            return "skip"

        with mock.patch.object(tuning, "reload_tuning_cell", fake_reload), mock.patch.object(
            tuning, "cell_disposition", fake_disposition
        ):
            with self.assertRaises(TuningStop) as caught:
                scan_block_cells(Path("."), "T1")
        self.assertIn("hash changed between cells", str(caught.exception))

    def test_attempt2_cells_do_not_use_the_preserved_paths(self):
        original = _cell_rels("T1", "FedNova", "0.1")[0]
        recovered = _cell_rels("T1", "FedNova", "0.1", attempt="attempt2")[0]
        self.assertNotEqual(original, recovered)
        self.assertIn("tuning/attempt2/cells/", recovered)
        self.assertNotIn("attempt2", original)

    def test_a_changed_preserved_cell_stops_the_recovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "q1_e5_results" / "tuning" / "cells"
            folder.mkdir(parents=True)
            target = folder / "T1_FedNova_lr0.1_s10007.json"
            target.write_text("{}\n", encoding="utf-8")
            snapshot = snapshot_original_attempt(root)
            assert_original_attempt_unchanged(root, snapshot)
            target.write_text('{"changed": true}\n', encoding="utf-8")
            with self.assertRaises(TuningStop) as caught:
                assert_original_attempt_unchanged(root, snapshot)
        self.assertIn("preserved 2026-10-06 tuning attempt was modified", str(caught.exception))

    def test_each_cell_reloads_when_that_cell_starts(self):
        import fl_local.e5_tuning as tuning

        calls = []

        def fake_reload(root, block):
            del root, block
            calls.append(len(calls))
            return self._bindings("index"), {0: np.array([len(calls)], dtype=np.int64)}

        def fake_disposition(root, block, arm, rate_token, bindings, attempt=None):
            del root, block, arm, rate_token, bindings, attempt
            return "run"

        with mock.patch.object(tuning, "reload_tuning_cell", fake_reload), mock.patch.object(
            tuning, "cell_disposition", fake_disposition
        ):
            first = tuning.load_cell_for_training(Path("."), "T1", "FedAvg", "0.001", attempt="attempt2")
            second = tuning.load_cell_for_training(Path("."), "T1", "FedNova", "0.001", attempt="attempt2")
        self.assertEqual(calls, [0, 1])
        self.assertNotEqual(int(first[2][0][0]), int(second[2][0][0]))
        tree = ast.parse(Path(tuning.__file__).read_text(encoding="utf-8"))
        func = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_run_tuning_blocks")
        found = False
        for rate_loop in ast.walk(func):
            if not isinstance(rate_loop, ast.For):
                continue
            if not (isinstance(rate_loop.iter, ast.Name) and rate_loop.iter.id == "TUNING_RATES"):
                continue
            for arm_loop in rate_loop.body:
                if not isinstance(arm_loop, ast.For):
                    continue
                if not (isinstance(arm_loop.iter, ast.Name) and arm_loop.iter.id == "ARMS"):
                    continue
                names = [
                    node.func.id
                    for node in ast.walk(arm_loop)
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                ]
                self.assertIn("load_cell_for_training", names)
                self.assertNotIn("scan_block_cells", names)
                found = True
        self.assertTrue(found)
        source = Path(tuning.__file__).read_text(encoding="utf-8")
        self.assertIn('return _run_tuning_blocks(root, ("T1",), attempt=ATTEMPT2, order_label="T1")', source)
        self.assertIs(_run_tuning_blocks.__defaults__, None)

    def test_this_process_is_visible_to_the_pid_check(self):
        import os

        self.assertTrue(_alive(os.getpid()))
        self.assertFalse(_alive(0))

    def test_attempt2_trains_t1_only(self):
        import fl_local.e5_tuning as tuning

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(TuningStop) as caught:
                _run_tuning_blocks(
                    Path(tmp),
                    ("T1", "T2"),
                    attempt="attempt2",
                    order_label="T1,T2",
                )
            self.assertIn("attempt2 trains T1 only", str(caught.exception))
            self.assertFalse((Path(tmp) / "q1_e5_results" / "tuning" / "attempt2" / "tune.pid").exists())
            self.assertFalse((Path(tmp) / "q1_e5_results" / "tuning" / "progress.log").exists())

    def test_plain_tune_does_not_modify_a_closed_preserved_attempt(self):
        import fl_local.e5_tuning as tuning

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "q1_e5_results" / "tuning" / "cells"
            folder.mkdir(parents=True)
            (folder / "T1_FedNova_lr0.1_s10007.json").write_text(
                json.dumps({"status": "failed", "finite": False}) + "\n",
                encoding="utf-8",
            )
            progress = root / "q1_e5_results" / "tuning" / "progress.log"
            progress.write_text("kept\n", encoding="utf-8")
            before = progress.read_bytes()
            with self.assertRaises(TuningStop) as caught:
                tuning.run_tuning(root)
            self.assertIn("plain tune will not resume it, replace it, or modify its files", str(caught.exception))
            self.assertIn("T1_FedNova_lr0.1_s10007.json", str(caught.exception))
            self.assertEqual(progress.read_bytes(), before)
            self.assertFalse((root / "q1_e5_results" / "tuning" / "tune.pid").exists())


def _fresh_preflight(**overrides) -> dict:
    report = {
        "accepted_prereg_sha256": ACCEPTED_PREREG_SHA256,
        "gpu_training": False,
        "kind": "no-training-preflight",
        "partition_files_written": False,
        "scalers_written": False,
        "status": "PASS",
        "written_utc": "2026-10-07T18:00:00Z",
    }
    report.update(overrides)
    return report


def _complete_decision_row(block="T1", seed=127, status="complete") -> dict:
    return {
        "Final_F1": "0.506024873091221",
        "arm": "CoordMedian",
        "block": block,
        "decision_cell": True,
        "finite": status == "complete",
        "kind": "decision",
        "learning_rate_value": 1e-3,
        "rounds": 30,
        "seed": seed,
        "status": status,
    }


class CoordMedianRunnerTests(unittest.TestCase):
    def test_plan_is_28_coordmedian_cells_starting_at_the_smoke_cell(self):
        plan = cell_plan()
        self.assertEqual(len(plan), 28)
        self.assertEqual(MEDIAN_FAMILY, FAMILY[:4])
        self.assertEqual([name for _, name in FAMILY[4:]], ["FedNova-FedAvg"] * 4)
        first = plan[0]
        self.assertEqual((first["block"], first["arm"], first["seed"], first["learning_rate_value"]), ("T1", "CoordMedian", 127, 1e-3))
        self.assertEqual([cell["block"] for cell in plan], [block for block in ("T1", "T2", "T5", "T6") for _ in range(7)])
        self.assertEqual([cell["seed"] for cell in plan[:7]], [127, 131, 137, 139, 149, 151, 157])
        self.assertTrue(all(cell["arm"] == "CoordMedian" and cell["optimizer"] == "adam" for cell in plan))

    def test_closed_commands_and_arms_are_refused_before_a_cache_load(self):
        for command in ("tune", "tune-t1-attempt2"):
            with self.assertRaises(runner.TrainingNotAuthorized) as caught:
                refuse_forbidden(command=command)
            self.assertIn("closed", str(caught.exception))
        with self.assertRaises(runner.TrainingNotAuthorized) as fednova:
            refuse_forbidden(arm="FedNova", block="T1", seed=127, learning_rate=1e-3)
        self.assertIn("FedNova", str(fednova.exception))
        with self.assertRaises(runner.TrainingNotAuthorized) as fedavg:
            refuse_forbidden(arm="FedAvg", block="T2", seed=131, learning_rate=1e-3)
        self.assertIn("FedAvg", str(fedavg.exception))
        with self.assertRaises(runner.TrainingNotAuthorized) as smoke:
            refuse_forbidden(arm="FedAvg", block="T1", seed=127, learning_rate=1e-3)
        self.assertIn("smoke cell", str(smoke.exception))
        with self.assertRaises(runner.TrainingNotAuthorized):
            refuse_forbidden(arm="CoordMedian", block="T1", seed=127, learning_rate=1e-2)
        with self.assertRaises(runner.TrainingNotAuthorized):
            refuse_forbidden(arm="CoordMedian", block="T1", seed=10007, learning_rate=1e-3)
        with self.assertRaises(runner.TrainingNotAuthorized):
            refuse_forbidden(arm="CoordMedian", block="T1", seed=42, learning_rate=1e-3)
        with mock.patch("fl_local.e5_coordmedian._load_sealed_cache", side_effect=AssertionError("cache")):
            with self.assertRaises(runner.TrainingNotAuthorized):
                train_coordmedian_cell(
                    ROOT,
                    {"arm": "FedNova", "block": "T1", "seed": 127, "learning_rate_value": 1e-3},
                )
            with self.assertRaises(runner.TrainingNotAuthorized):
                train_coordmedian_cell(
                    ROOT,
                    {"arm": "FedAvg", "block": "T1", "seed": 127, "learning_rate_value": 1e-3},
                )

    def test_trainer_source_calls_only_the_coordmedian_clone(self):
        source = Path(train_coordmedian_cell.__code__.co_filename).read_text(encoding="utf-8")
        tree = ast.parse(source)
        called = set()
        clone_calls = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else ""
            called.add(name)
            if name == "run_match_notebook_clone":
                clone_calls.append(node)
        for forbidden in (
            "run_tuning",
            "run_t1_attempt2",
            "momentum_sgd_client_update",
            "fednova_server_update",
            "fedavg_server_update",
            "use_deterministic_algorithms",
        ):
            self.assertNotIn(forbidden, called)
        self.assertEqual(len(clone_calls), 1)
        algorithm = next(keyword for keyword in clone_calls[0].keywords if keyword.arg == "algorithm")
        self.assertEqual(algorithm.value.value, "CoordMedian")

    def test_holm_controls_only_the_four_median_tests(self):
        raw = {
            ("T1", "FT-CoordMedian"): 0.04,
            ("T2", "FT-CoordMedian"): 0.01,
            ("T5", "FT-CoordMedian"): 0.03,
            ("T6", "FT-CoordMedian"): 0.02,
        }
        adjusted = adjust_median_family(raw)
        self.assertAlmostEqual(adjusted[("T2", "FT-CoordMedian")], 0.04)
        self.assertAlmostEqual(adjusted[("T6", "FT-CoordMedian")], 0.06)
        self.assertAlmostEqual(adjusted[("T5", "FT-CoordMedian")], 0.06)
        self.assertAlmostEqual(adjusted[("T1", "FT-CoordMedian")], 0.06)
        capped = holm_four({key: 0.4 for key in MEDIAN_FAMILY})
        self.assertTrue(all(value == 1.0 for value in capped.values()))
        with self.assertRaises(ValueError):
            holm_four({key: 0.2 for key in FAMILY})
        with self.assertRaises(runner.TrainingNotAuthorized):
            adjust_median_family({("T2", "FA-UW"): 0.01, **{key: 0.2 for key in MEDIAN_FAMILY}})
        with self.assertRaises(runner.TrainingNotAuthorized):
            adjust_median_family({key: 0.2 for key in FAMILY})

    def test_stale_preflight_and_the_remaining_cells_do_not_start(self):
        stale = _fresh_preflight(written_utc=STALE_PREFLIGHT_UTC)
        with self.assertRaises(runner.TrainingNotAuthorized) as caught:
            assert_fresh_preflight(stale)
        self.assertIn("predates", str(caught.exception))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(runner.TrainingNotAuthorized):
                run_coordmedian(root, report=stale, limit=1)
            with self.assertRaises(runner.TrainingNotAuthorized) as remaining:
                run_coordmedian(root, report=_fresh_preflight(), limit=None, authorize_remaining=False)
            self.assertIn("not authorized", str(remaining.exception))
            self.assertFalse((root / "q1_e5_results").exists())

    def test_authorized_plan_uses_the_fake_trainer_and_skips_a_paused_cell(self):
        calls = []

        def fake(_root, cell):
            calls.append((cell["block"], cell["arm"], cell["seed"], cell["learning_rate_value"]))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            smoke = run_coordmedian(root, report=_fresh_preflight(), train_cell=fake)
            self.assertEqual(calls, [("T1", "CoordMedian", 127, 1e-3)])
            self.assertEqual(smoke["limit"], 1)
            self.assertFalse(smoke["authorize_remaining"])
            self.assertFalse((root / "q1_e5_results" / "tuning").exists())
            self.assertFalse((root / "q1_e5_results" / "partitions").exists())
            self.assertFalse((root / "q1_e5_results" / "coordmedian" / "coordmedian.pid").exists())
            paused = root / "q1_e5_results" / "coordmedian" / "cells" / "T1_CoordMedian_s127.json"
            paused.parent.mkdir(parents=True, exist_ok=True)
            paused.write_text(json.dumps({"status": "failed", "finite": False}) + "\n", encoding="utf-8")
            before = list(calls)
            with self.assertRaises(CoordMedianStop) as caught:
                run_coordmedian(root, report=_fresh_preflight(), train_cell=fake)
            self.assertIn("will not be replaced", str(caught.exception))
            self.assertEqual(calls, before)

        calls.clear()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            done = run_coordmedian(
                root,
                report=_fresh_preflight(),
                limit=None,
                authorize_remaining=True,
                train_cell=fake,
            )
        self.assertEqual(done["planned"], 28)
        self.assertEqual(len(calls), 28)
        self.assertEqual(calls[0], ("T1", "CoordMedian", 127, 1e-3))
        self.assertTrue(all(arm == "CoordMedian" and rate == 1e-3 for _, arm, _, rate in calls))

        def run_tuning(*_args, **_kwargs):
            raise AssertionError("tuned")

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(runner.TrainingNotAuthorized):
                run_coordmedian(
                    Path(tmp),
                    report=_fresh_preflight(),
                    train_cell=run_tuning,
                )

    def test_coordmedian_command_requires_a_fresh_preflight_and_starts_one_cell(self):
        self.assertEqual(runner.required_partition_rels("coordmedian"), ())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected = {}
            for relative in BOUND_SHA256:
                payload = f"bound:{relative}".encode()
                _write_bytes(root / relative, payload)
                expected[relative] = hashlib.sha256(payload).hexdigest()
            receipt = runner.section_91_recheck(
                root,
                command="coordmedian",
                expected_hashes=expected,
                check_environment=False,
                apply_cuda=False,
            )
            self.assertEqual(receipt.command, "coordmedian")
            self.assertFalse(any((root / relative).exists() for relative in (
                "q1_e5_results/partitions/decision_CIC-IoT2023_s127.npz",
                "q1_e5_results/partitions/decision_N-BaIoT_device.npz",
            )))
        stale = _fresh_preflight(written_utc=STALE_PREFLIGHT_UTC)
        with mock.patch("run_q1_e5.load_preflight_report", return_value=stale), mock.patch(
            "run_q1_e5.run_coordmedian", side_effect=AssertionError("trained")
        ), mock.patch("fl_local.e5_coordmedian.train_coordmedian_cell", side_effect=AssertionError("gpu")):
            with self.assertRaises(runner.TrainingNotAuthorized):
                runner.main(["coordmedian"])
        with self.assertRaises(runner.TrainingNotAuthorized) as waved:
            runner.main(["coordmedian", "--wave", "0"])
        self.assertIn("FedAvg", str(waved.exception))
        fresh = _fresh_preflight()
        with mock.patch("run_q1_e5.load_preflight_report", return_value=fresh), mock.patch(
            "run_q1_e5.section_91_recheck", return_value=runner.RecheckReceipt("coordmedian")
        ), mock.patch("run_q1_e5.run_coordmedian", return_value={"status": "mocked"}) as started, mock.patch(
            "fl_local.e5_coordmedian.train_coordmedian_cell", side_effect=AssertionError("gpu")
        ):
            self.assertEqual(runner.main(["coordmedian"]), 0)
        started.assert_called_once()
        self.assertEqual(started.call_args.kwargs["limit"], 1)
        self.assertFalse(started.call_args.kwargs["authorize_remaining"])
        source = Path(runner.__file__).read_text(encoding="utf-8")
        main_source = source[source.index("def main") :]
        self.assertIn("limit=1", main_source)
        self.assertIn("authorize_remaining=False", main_source)
        self.assertNotIn("run_tuning(", main_source)
        self.assertNotIn("run_t1_attempt2(", main_source)

    def test_continuation_skips_only_a_complete_smoke_cell(self):
        calls = []

        def fake(_root, cell):
            calls.append((cell["block"], cell["arm"], cell["seed"], cell["learning_rate_value"]))

        def write_row(root, name, row):
            path = root / "q1_e5_results" / "coordmedian" / "cells" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(runner.TrainingNotAuthorized) as missing:
                run_coordmedian(
                    root,
                    report=_fresh_preflight(),
                    limit=None,
                    authorize_remaining=True,
                    continuation=True,
                    train_cell=fake,
                )
            self.assertIn("not complete", str(missing.exception))
            self.assertEqual(calls, [])
            self.assertFalse((root / "q1_e5_results" / "coordmedian").exists())

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_row(root, "T1_CoordMedian_s127.json", _complete_decision_row(status="failed"))
            with self.assertRaises(CoordMedianStop) as stopped:
                run_coordmedian(
                    root,
                    report=_fresh_preflight(),
                    limit=None,
                    authorize_remaining=True,
                    continuation=True,
                    train_cell=fake,
                )
            self.assertIn("will not be replaced", str(stopped.exception))
            self.assertEqual(calls, [])
            self.assertFalse((root / "q1_e5_results" / "coordmedian" / "coordmedian.pid").exists())

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_row(root, "T1_CoordMedian_s127.json", _complete_decision_row())
            with self.assertRaises(runner.TrainingNotAuthorized) as narrow:
                run_coordmedian(
                    root,
                    report=_fresh_preflight(),
                    limit=1,
                    authorize_remaining=True,
                    continuation=True,
                    train_cell=fake,
                )
            self.assertIn("authorized continuation", str(narrow.exception))
            write_row(root, "T1_CoordMedian_s131.json", _complete_decision_row(seed=131, status="running"))
            with self.assertRaises(CoordMedianStop) as paused_later:
                run_coordmedian(
                    root,
                    report=_fresh_preflight(),
                    limit=None,
                    authorize_remaining=True,
                    continuation=True,
                    train_cell=fake,
                )
            self.assertIn("T1 CoordMedian seed 131", str(paused_later.exception))
            self.assertEqual(calls, [])
            self.assertFalse((root / "q1_e5_results" / "coordmedian" / "coordmedian.pid").exists())

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_row(root, "T1_CoordMedian_s127.json", _complete_decision_row())
            result = run_coordmedian(
                root,
                report=_fresh_preflight(),
                limit=None,
                authorize_remaining=True,
                continuation=True,
                train_cell=fake,
            )
            self.assertTrue(result["continuation"])
            self.assertTrue(result["authorize_remaining"])
            self.assertIsNone(result["limit"])
            self.assertEqual(result["started_cells"][0], {"block": "T1", "seed": 127, "status": "skip"})
            self.assertEqual(len(calls), 27)
            self.assertEqual(calls[0], ("T1", "CoordMedian", 131, 1e-3))
            self.assertNotIn(("T1", "CoordMedian", 127, 1e-3), calls)
            self.assertEqual([seed for block, _, seed, _ in calls if block == "T1"], [131, 137, 139, 149, 151, 157])
            self.assertEqual([block for block, _, seed, _ in calls if seed == 127], ["T2", "T5", "T6"])
            self.assertTrue(all(arm == "CoordMedian" and rate == 1e-3 for _, arm, _, rate in calls))
            self.assertFalse((root / "q1_e5_results" / "tuning").exists())
            self.assertFalse((root / "q1_e5_results" / "partitions").exists())
            self.assertFalse((root / "q1_e5_results" / "coordmedian" / "coordmedian.pid").exists())

    def test_continue_command_rechecks_and_does_not_train_the_smoke_cell(self):
        self.assertEqual(runner.required_partition_rels("coordmedian-continue"), ())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected = {}
            for relative in BOUND_SHA256:
                payload = f"bound:{relative}".encode()
                _write_bytes(root / relative, payload)
                expected[relative] = hashlib.sha256(payload).hexdigest()
            receipt = runner.section_91_recheck(
                root,
                command="coordmedian-continue",
                expected_hashes=expected,
                check_environment=False,
                apply_cuda=False,
            )
            self.assertEqual(receipt.command, "coordmedian-continue")
            self.assertFalse(any((root / relative).exists() for relative in (
                "q1_e5_results/partitions/decision_CIC-IoT2023_s127.npz",
                "q1_e5_results/partitions/decision_N-BaIoT_device.npz",
            )))
        stale = _fresh_preflight(written_utc=STALE_PREFLIGHT_UTC)
        with mock.patch("run_q1_e5.load_preflight_report", return_value=stale), mock.patch(
            "run_q1_e5.run_coordmedian", side_effect=AssertionError("trained")
        ), mock.patch("fl_local.e5_coordmedian.train_coordmedian_cell", side_effect=AssertionError("gpu")):
            with self.assertRaises(runner.TrainingNotAuthorized):
                runner.main(["coordmedian-continue"])
        with self.assertRaises(runner.TrainingNotAuthorized) as waved:
            runner.main(["coordmedian-continue", "--wave", "0"])
        self.assertIn("FedAvg", str(waved.exception))
        fresh = _fresh_preflight()
        with mock.patch("run_q1_e5.load_preflight_report", return_value=fresh), mock.patch(
            "run_q1_e5.section_91_recheck", return_value=runner.RecheckReceipt("coordmedian-continue")
        ), mock.patch("run_q1_e5.run_coordmedian", return_value={"status": "mocked"}) as started, mock.patch(
            "fl_local.e5_coordmedian.train_coordmedian_cell", side_effect=AssertionError("gpu")
        ):
            self.assertEqual(runner.main(["coordmedian-continue"]), 0)
        started.assert_called_once()
        self.assertIsNone(started.call_args.kwargs["limit"])
        self.assertTrue(started.call_args.kwargs["authorize_remaining"])
        self.assertTrue(started.call_args.kwargs["continuation"])


if __name__ == "__main__":
    unittest.main()
