"""Q1-E5 local-update protocol.

CoordMedian calls the unmodified engine.train_client_nb path. FedNova and
matched FedAvg call momentum_sgd_client_update. engine.py is not modified.
"""
from __future__ import annotations

import ast
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import fl_local.engine as engine
from fl_local.e5_aggregators import (
    CUDA_FLAGS,
    GENERATOR_STRIDE,
    coordinate_median,
    expected_local_steps,
    normalizing_constant,
    require_finite,
)

PROTOCOL_PATH = Path(__file__).resolve()


def _assert_deterministic_algorithms_not_called() -> None:
    """Reject a call. A mention in this check's own message is not a call."""
    tree = ast.parse(PROTOCOL_PATH.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "use_deterministic_algorithms":
            raise RuntimeError("E5 must not call torch.use_deterministic_algorithms")
        if isinstance(node, ast.Name) and node.id == "use_deterministic_algorithms":
            raise RuntimeError("E5 must not call torch.use_deterministic_algorithms")


def apply_and_read_cuda_flags() -> dict:
    """Set the four frozen flags, then read them back. Do not trust a default."""
    _assert_deterministic_algorithms_not_called()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    observed = {
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
    }
    if observed != CUDA_FLAGS:
        raise RuntimeError(f"CUDA flag mismatch: {observed}")
    return observed


def client_generator(seed: int, client_index: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(int(seed) + GENERATOR_STRIDE * (int(client_index) + 1))
    return generator


def _loader(x_local: np.ndarray, y_local: np.ndarray, batch_size: int, generator: torch.Generator):
    dataset = TensorDataset(
        torch.tensor(np.ascontiguousarray(x_local), dtype=torch.float32),
        torch.tensor(np.ascontiguousarray(y_local), dtype=torch.long),
    )
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=True,
        drop_last=False,
        generator=generator,
    )


def make_momentum_sgd(model: torch.nn.Module, lr: float) -> torch.optim.Optimizer:
    return torch.optim.SGD(
        model.parameters(),
        lr=float(lr),
        momentum=0.9,
        dampening=0.0,
        nesterov=False,
        weight_decay=0.0,
    )


def assert_client_endpoint(start: np.ndarray, delta: np.ndarray, endpoint: np.ndarray) -> None:
    """Require a finite float32 endpoint to equal start plus delta_i.

    The sum is formed in float64. Every finite float32 value is exact in
    float64, so the identity holds at large finite magnitudes. A float32
    subtraction of those same values need not. Non-finite values are
    rejected before equality, because NaN compares unequal to itself.
    """
    start_array = np.asarray(start)
    delta_array = np.asarray(delta)
    endpoint_array = np.asarray(endpoint)
    if start_array.shape != endpoint_array.shape or delta_array.shape != endpoint_array.shape:
        raise RuntimeError("endpoint is not the start plus delta_i in float32")
    finite = (
        np.all(np.isfinite(start_array))
        and np.all(np.isfinite(delta_array))
        and np.all(np.isfinite(endpoint_array))
    )
    if not finite:
        raise RuntimeError("delta_i is non-finite")
    rebuilt = start_array.astype(np.float64) + delta_array.astype(np.float64)
    if not np.array_equal(rebuilt.astype(np.float32), np.asarray(endpoint_array, dtype=np.float32)):
        raise RuntimeError("endpoint is not the start plus delta_i in float32")


def momentum_sgd_client_update(
    start: Sequence[np.ndarray],
    x_local: np.ndarray,
    y_local: np.ndarray,
    *,
    lr: float,
    generator: torch.Generator,
    model_name: str,
    num_features: int,
    num_classes: int,
    device: torch.device,
    batch_size: int = 512,
    local_epochs: int = 3,
) -> dict:
    """One client round. A new optimizer starts with empty momentum state.

    Returns the endpoint, delta_i, and scalar A_i. The endpoint equals the
    starting parameters plus delta_i.
    """
    if generator is None:
        raise ValueError("the SGD client update requires the persistent generator")
    start_arrays = [np.asarray(layer, dtype=np.float32).copy() for layer in start]
    for layer in start_arrays:
        require_finite(layer, "starting parameters")
    model = engine.make_model(model_name, num_features, num_classes, device)
    engine.set_parameters_np(model, start_arrays)
    optimizer = make_momentum_sgd(model, lr)
    state_at_start = len(optimizer.state)
    if state_at_start != 0:
        raise RuntimeError("momentum state must be empty when the client round starts")
    loader = _loader(x_local, y_local, batch_size, generator)
    criterion = torch.nn.CrossEntropyLoss()
    steps = 0
    for _ in range(int(local_epochs)):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            steps += 1
    expected = expected_local_steps(len(y_local), epochs=local_epochs, batch_size=batch_size)
    if steps != expected:
        raise RuntimeError(f"counted optimizer.step() calls {steps} != {expected}")
    area = normalizing_constant(steps)
    endpoint = [layer.detach().cpu().numpy().astype(np.float32, copy=True) for layer in model.state_dict().values()]
    delta = [
        endpoint_layer.astype(np.float64) - start_layer.astype(np.float64)
        for endpoint_layer, start_layer in zip(endpoint, start_arrays)
    ]
    for layer_index, layer in enumerate(delta):
        assert_client_endpoint(start_arrays[layer_index], layer, endpoint[layer_index])
    return {
        "endpoint": endpoint,
        "delta": delta,
        "A_i": area,
        "steps": steps,
        "momentum_state_at_start": state_at_start,
    }


def minibatch_sequences(
    n_rows: int,
    *,
    seed: int,
    client_index: int,
    rounds: int,
    batch_size: int,
    persistent: bool,
) -> list[list[int]]:
    """Record label order. persistent=False recreates the client seed every round."""
    labels = np.arange(n_rows, dtype=np.int64)
    features = np.zeros((n_rows, 1), dtype=np.float32)
    sequences = []
    generator = client_generator(seed, client_index) if persistent else None
    for _ in range(int(rounds)):
        if not persistent:
            generator = client_generator(seed, client_index)
        loader = _loader(features, labels, batch_size, generator)
        order = []
        for _, yb in loader:
            order.extend(int(value) for value in yb.numpy())
        sequences.append(order)
    return sequences


def run_match_notebook_clone(
    *,
    model_name: str,
    algorithm: str,
    partition: dict,
    seed: int,
    x_train,
    y_train,
    test_loader: DataLoader,
    num_features: int,
    num_classes: int,
    device: torch.device,
    cfg: dict,
    verbose: bool = False,
    on_round=None,
) -> dict:
    """E4 match_notebook order. Only the server aggregation can differ.

    FedTrimmedAvg uses the sealed trimmed-mean function. CoordMedian uses the
    fifth ordered value. The local step is engine.train_client_nb, which builds
    DataLoader(..., shuffle=True) with no generator argument.
    """
    if algorithm not in {"FedTrimmedAvg", "CoordMedian"}:
        raise ValueError(f"clone aggregation must be FedTrimmedAvg or CoordMedian, got {algorithm}")
    engine.set_global_seed(int(seed))
    x_np, y_np = engine._as_numpy_xy(x_train, y_train)
    num_clients = int(cfg["num_clients"])
    num_rounds = int(cfg["num_rounds"])
    local_epochs = int(cfg["local_epochs"])
    batch_size = int(cfg["batch_size"])
    lr = float(cfg["lr"])
    global_model = engine.make_model(model_name, num_features, num_classes, device)
    global_params = engine.get_parameters_np(global_model)
    loss0, acc0, f10 = engine.evaluate_model_nb(global_model, test_loader, device)
    if verbose:
        print(
            f"  Round  0 (init) | Acc {acc0:.4f} F1 {f10:.4f} Loss {loss0:.4f}  "
            f"[{algorithm} seed={seed} mode=match_notebook]"
        )
    history = []
    for rnd in range(1, num_rounds + 1):
        client_results = []
        for cid in range(num_clients):
            idx = partition[cid]
            if len(idx) == 0:
                continue
            updated, n_samp = engine.train_client_nb(
                global_params,
                x_np[idx],
                y_np[idx],
                local_epochs=local_epochs,
                lr=lr,
                batch_size=batch_size,
                device=device,
                model_name=model_name,
                num_features=num_features,
                num_classes=num_classes,
                proximal_mu=0.0,
                optimizer_name=cfg.get("optimizer", "adam"),
                sgd_momentum=float(cfg.get("sgd_momentum", 0.9)),
            )
            client_results.append((updated, n_samp))
        client_params_list = [params for params, _ in client_results]
        if algorithm == "FedTrimmedAvg":
            global_params = engine.aggregate_trimmed_avg_np(
                client_params_list, cfg["trimmed_fraction"]
            )
        else:
            global_params = coordinate_median(client_params_list)
        engine.set_parameters_np(global_model, global_params)
        loss, acc, f1 = engine.evaluate_model_nb(global_model, test_loader, device)
        item = {"round": rnd, "accuracy": acc, "f1": f1, "loss": loss}
        history.append(item)
        if on_round is not None:
            on_round(dict(item))
    return {
        "history": history,
        "parameters": [np.asarray(layer).copy() for layer in global_params],
    }


def clone_source_calls_train_client_nb() -> bool:
    """The clone must call train_client_nb and must not pass a DataLoader generator."""
    tree = ast.parse(PROTOCOL_PATH.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or node.name != "run_match_notebook_clone":
            continue
        found_call = False
        for child in ast.walk(node):
            if not isinstance(child, ast.Call):
                continue
            func = child.func
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            else:
                name = ""
            if name == "DataLoader":
                return False
            if any(keyword.arg == "generator" for keyword in child.keywords):
                return False
            if name == "train_client_nb":
                found_call = True
        return found_call
    return False


def initial_parameter_bytes(seed: int, num_features: int, num_classes: int, device: torch.device):
    engine.set_global_seed(int(seed))
    model = engine.make_model("MLP", num_features, num_classes, device)
    return engine.get_parameters_np(model)


# FedNova and matched FedAvg call this one object. There is no second local loop.
FEDNOVA_CLIENT_UPDATE = momentum_sgd_client_update
FEDAVG_CLIENT_UPDATE = momentum_sgd_client_update
