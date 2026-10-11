"""
FL engine for local Q1-E* work.

Modes
-----
match_notebook
  Faithful to the paper notebook / run_hardware_probe.py path:
  - numpy parameter lists
  - new Module() per client (init burns RNG, then set_parameters)
  - DataLoader(..., shuffle=True) + train_one_epoch as in the notebook
  - numpy FedAvg / unweighted mean / trimmed mean
  Use this for any number that must sit next to published tables.

fast
  Throughput path (reuse client module, torch params, manual batches).
  Internally consistent only — do not mix with published cells.
"""
from __future__ import annotations

import random
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

ArrayOrTensor = Union[np.ndarray, torch.Tensor]

CFG_FULL = dict(
    max_samples_per_file=20000,
    num_rounds=30,
    local_epochs=3,
    batch_size=512,
    lr=1e-3,
    test_fraction=0.2,
    fedprox_mu=0.01,
    # Q1-E3 knobs — defaults preserve published / pre-E3 behaviour
    optimizer="adam",  # adam | sgd
    sgd_momentum=0.9,  # used only when optimizer == sgd (prereg r3)
    trimmed_fraction=0.1,
    num_clients=9,
    seeds=[42, 123, 7],
    models=["MLP", "CNN"],
    dirichlet_alphas=[1.0, 0.5, 0.1],
)


def make_optimizer(
    model: nn.Module,
    lr: float,
    optimizer_name: str = "adam",
    sgd_momentum: float = 0.9,
) -> optim.Optimizer:
    """Build client optimizer. Default adam matches pre-E3 hardcoding."""
    name = (optimizer_name or "adam").strip().lower()
    if name == "adam":
        return optim.Adam(model.parameters(), lr=lr)
    if name == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=float(sgd_momentum),
            weight_decay=0.0,
        )
    raise ValueError(f"Unknown optimizer: {optimizer_name!r} (use adam|sgd)")



def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def enable_repro_flags() -> None:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class IoTBotnetMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Dropout(0.2),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.LayerNorm(32),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class IoTBotnetCNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.reshape = nn.Unflatten(1, (1, input_dim))
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.LayerNorm(32),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        x = self.reshape(x)
        x = self.features(x)
        return self.classifier(x)


def make_model(name: str, num_features: int, num_classes: int, device: torch.device):
    if name == "MLP":
        m = IoTBotnetMLP(num_features, num_classes)
    elif name == "CNN":
        m = IoTBotnetCNN(num_features, num_classes)
    else:
        raise ValueError(name)
    return m.to(device)


# ---- notebook-faithful param I/O (numpy) ---------------------------------


def get_parameters_np(model: nn.Module) -> List[np.ndarray]:
    return [val.cpu().numpy() for val in model.state_dict().values()]


def set_parameters_np(model: nn.Module, parameters: Sequence[np.ndarray]) -> None:
    state_dict = OrderedDict(
        {
            k: torch.tensor(v, dtype=torch.float32)
            for k, v in zip(model.state_dict().keys(), parameters)
        }
    )
    model.load_state_dict(state_dict, strict=True)


def get_parameters_torch(model: nn.Module) -> List[torch.Tensor]:
    return [p.detach().float().cpu().clone() for p in model.state_dict().values()]


def set_parameters_torch(model: nn.Module, parameters: Sequence[torch.Tensor]) -> None:
    state = OrderedDict(
        {
            k: v.detach().to(dtype=torch.float32)
            for k, v in zip(model.state_dict().keys(), parameters)
        }
    )
    model.load_state_dict(state, strict=True)


def partition_dirichlet(y_np: np.ndarray, num_clients: int, alpha: float, seed: int):
    rng = np.random.RandomState(seed)
    num_classes = len(np.unique(y_np))
    client_indices = {cid: [] for cid in range(num_clients)}
    for c in range(num_classes):
        class_idx = np.where(y_np == c)[0]
        rng.shuffle(class_idx)
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        proportions = np.maximum(proportions, 1e-6)
        proportions = proportions / proportions.sum()
        splits = (np.cumsum(proportions) * len(class_idx)).astype(int)[:-1]
        for cid, chunk in enumerate(np.split(class_idx, splits)):
            client_indices[cid].extend(chunk.tolist())
    for cid in range(num_clients):
        if len(client_indices[cid]) == 0:
            donor = max(range(num_clients), key=lambda c: len(client_indices[c]))
            if client_indices[donor]:
                client_indices[cid].append(client_indices[donor].pop())
    for cid in range(num_clients):
        rng.shuffle(client_indices[cid])
        client_indices[cid] = np.array(client_indices[cid], dtype=np.int64)
    return client_indices


def partition_iid(n: int, num_clients: int, seed: int):
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)
    splits = np.array_split(indices, num_clients)
    return {i: s.astype(np.int64) for i, s in enumerate(splits)}


# ---- aggregators (numpy — notebook) --------------------------------------


def aggregate_fedavg_np(
    client_params_list: List[List[np.ndarray]], client_sizes: List[int]
) -> List[np.ndarray]:
    total = sum(client_sizes)
    weights = [n / total for n in client_sizes]
    num_layers = len(client_params_list[0])
    aggregated = []
    for layer_i in range(num_layers):
        weighted_sum = np.zeros_like(client_params_list[0][layer_i])
        for c, w in enumerate(weights):
            weighted_sum += w * client_params_list[c][layer_i]
        aggregated.append(weighted_sum)
    return aggregated


def aggregate_unweighted_mean_np(
    client_params_list: List[List[np.ndarray]],
) -> List[np.ndarray]:
    n = float(len(client_params_list))
    num_layers = len(client_params_list[0])
    out = []
    for layer_i in range(num_layers):
        acc = np.zeros_like(client_params_list[0][layer_i])
        for c in range(len(client_params_list)):
            acc = acc + client_params_list[c][layer_i]
        out.append(acc / n)
    return out


def aggregate_trimmed_avg_np(
    client_params_list: List[List[np.ndarray]], trim_fraction: float = 0.1
) -> List[np.ndarray]:
    n_clients = len(client_params_list)
    trim_n = max(1, int(trim_fraction * n_clients))
    num_layers = len(client_params_list[0])
    aggregated = []
    for layer_i in range(num_layers):
        stacked = np.stack([c[layer_i] for c in client_params_list], axis=0)
        sorted_v = np.sort(stacked, axis=0)
        if n_clients > 2 * trim_n:
            trimmed = sorted_v[trim_n:-trim_n]
        else:
            trimmed = sorted_v
        aggregated.append(np.mean(trimmed, axis=0))
    return aggregated


def aggregate_weighted_trimmed_avg_np(
    client_params_list: List[List[np.ndarray]],
    client_sizes: List[int],
    trim_fraction: float = 0.1,
) -> List[np.ndarray]:
    """Coordinate-wise trim, then size-weighted mean of remaining clients (Q1-E1b).

    For each coordinate: sort clients by value, drop lowest/highest ``trim_n``
    clients, then weight the rest by client sample counts (renormalized).
    When all sizes are equal this matches unweighted ``FedTrimmedAvg``.
    """
    n_clients = len(client_params_list)
    if n_clients == 0:
        raise ValueError("empty client_params_list")
    if len(client_sizes) != n_clients:
        raise ValueError("client_sizes length must match client_params_list")
    trim_n = max(1, int(trim_fraction * n_clients))
    weights = np.asarray(client_sizes, dtype=np.float64)
    if np.any(weights < 0) or float(weights.sum()) <= 0:
        raise ValueError("client_sizes must be non-negative with positive sum")
    out: List[np.ndarray] = []
    for layer_i in range(len(client_params_list[0])):
        stacked = np.stack([c[layer_i] for c in client_params_list], axis=0)
        if n_clients <= 2 * trim_n:
            w = weights / weights.sum()
            acc = np.zeros_like(stacked[0], dtype=np.float64)
            for c, wc in enumerate(w):
                acc = acc + wc * stacked[c]
            out.append(acc.astype(stacked.dtype, copy=False))
            continue
        order = np.argsort(stacked, axis=0)
        sorted_vals = np.take_along_axis(stacked, order, axis=0)
        w_exp = weights.reshape((n_clients,) + (1,) * (stacked.ndim - 1))
        w_b = np.broadcast_to(w_exp, stacked.shape)
        sorted_w = np.take_along_axis(w_b, order, axis=0)
        keep_v = sorted_vals[trim_n:-trim_n]
        keep_w = sorted_w[trim_n:-trim_n].astype(np.float64, copy=False)
        w_sum = keep_w.sum(axis=0)
        w_sum = np.maximum(w_sum, 1e-12)
        agg = (keep_v.astype(np.float64, copy=False) * keep_w).sum(axis=0) / w_sum
        out.append(agg.astype(stacked.dtype, copy=False))
    return out


# ---- torch aggregators (fast mode only) ----------------------------------


def aggregate_fedavg_torch(
    client_params_list: List[List[torch.Tensor]], client_sizes: List[int]
) -> List[torch.Tensor]:
    total = float(sum(client_sizes))
    weights = [n / total for n in client_sizes]
    out = []
    for layer_i in range(len(client_params_list[0])):
        acc = torch.zeros_like(client_params_list[0][layer_i])
        for c, w in enumerate(weights):
            acc = acc + w * client_params_list[c][layer_i]
        out.append(acc)
    return out


def aggregate_unweighted_mean_torch(
    client_params_list: List[List[torch.Tensor]],
) -> List[torch.Tensor]:
    n = float(len(client_params_list))
    out = []
    for layer_i in range(len(client_params_list[0])):
        acc = torch.zeros_like(client_params_list[0][layer_i])
        for c in range(len(client_params_list)):
            acc = acc + client_params_list[c][layer_i]
        out.append(acc / n)
    return out


def aggregate_trimmed_avg_torch(
    client_params_list: List[List[torch.Tensor]], trim_fraction: float = 0.1
) -> List[torch.Tensor]:
    n_clients = len(client_params_list)
    trim_n = max(1, int(trim_fraction * n_clients))
    out = []
    for layer_i in range(len(client_params_list[0])):
        stacked = torch.stack([c[layer_i] for c in client_params_list], dim=0)
        sorted_v, _ = torch.sort(stacked, dim=0)
        trimmed = sorted_v[trim_n:-trim_n] if n_clients > 2 * trim_n else sorted_v
        out.append(trimmed.mean(dim=0))
    return out


def aggregate_weighted_trimmed_avg_torch(
    client_params_list: List[List[torch.Tensor]],
    client_sizes: List[int],
    trim_fraction: float = 0.1,
) -> List[torch.Tensor]:
    """Torch twin of ``aggregate_weighted_trimmed_avg_np`` (fast mode only)."""
    n_clients = len(client_params_list)
    if n_clients == 0:
        raise ValueError("empty client_params_list")
    if len(client_sizes) != n_clients:
        raise ValueError("client_sizes length must match client_params_list")
    trim_n = max(1, int(trim_fraction * n_clients))
    weights = torch.tensor(
        client_sizes, dtype=torch.float64, device=client_params_list[0][0].device
    )
    if torch.any(weights < 0) or float(weights.sum()) <= 0:
        raise ValueError("client_sizes must be non-negative with positive sum")
    out: List[torch.Tensor] = []
    for layer_i in range(len(client_params_list[0])):
        stacked = torch.stack([c[layer_i] for c in client_params_list], dim=0)
        if n_clients <= 2 * trim_n:
            w = (weights / weights.sum()).to(dtype=stacked.dtype)
            acc = torch.zeros_like(stacked[0])
            for c in range(n_clients):
                acc = acc + w[c] * stacked[c]
            out.append(acc)
            continue
        sorted_vals, order = torch.sort(stacked, dim=0)
        w_exp = weights.reshape((n_clients,) + (1,) * (stacked.ndim - 1))
        w_b = w_exp.expand_as(stacked)
        sorted_w = torch.gather(w_b, 0, order)
        keep_v = sorted_vals[trim_n:-trim_n]
        keep_w = sorted_w[trim_n:-trim_n].to(dtype=torch.float64)
        w_sum = keep_w.sum(dim=0).clamp_min(1e-12)
        agg = (keep_v.to(dtype=torch.float64) * keep_w).sum(dim=0) / w_sum
        out.append(agg.to(dtype=stacked.dtype))
    return out


# aliases used by older imports
aggregate_fedavg = aggregate_fedavg_np
aggregate_unweighted_mean = aggregate_unweighted_mean_np
aggregate_trimmed_avg = aggregate_trimmed_avg_np
aggregate_weighted_trimmed_avg = aggregate_weighted_trimmed_avg_np


def train_one_epoch_nb(
    model, loader, optimizer, criterion, device, proximal_mu=0.0, global_params=None
):
    """Notebook-identical local epoch."""
    model.train()
    total_loss, total_n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        if proximal_mu > 0.0 and global_params is not None:
            prox = sum(
                ((lp - gp) ** 2).sum()
                for lp, gp in zip(model.parameters(), global_params)
            )
            loss = loss + (proximal_mu / 2.0) * prox
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(yb)
        total_n += len(yb)
    return total_loss / max(total_n, 1)


def evaluate_model_nb(model, loader, device):
    """Notebook-identical evaluation (torch.no_grad + list extend)."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    preds_all, labels_all = [], []
    total_loss, total_n = 0.0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            total_loss += criterion(out, yb).item() * len(yb)
            total_n += len(yb)
            preds_all.extend(out.argmax(1).cpu().numpy())
            labels_all.extend(yb.cpu().numpy())
    acc = accuracy_score(labels_all, preds_all)
    f1 = f1_score(labels_all, preds_all, average="macro", zero_division=0)
    return total_loss / max(total_n, 1), acc, f1


def train_client_nb(
    global_params: List[np.ndarray],
    X_local: np.ndarray,
    y_local: np.ndarray,
    local_epochs: int,
    lr: float,
    batch_size: int,
    device: torch.device,
    model_name: str,
    num_features: int,
    num_classes: int,
    proximal_mu: float = 0.0,
    optimizer_name: str = "adam",
    sgd_momentum: float = 0.9,
):
    """Notebook train_client: new model, TensorDataset, DataLoader shuffle=True."""
    model = make_model(model_name, num_features, num_classes, device)
    set_parameters_np(model, global_params)

    if proximal_mu > 0:
        global_param_tensors = [p.clone().detach() for p in model.parameters()]
    else:
        global_param_tensors = None

    dataset = TensorDataset(
        torch.tensor(X_local, dtype=torch.float32),
        torch.tensor(y_local, dtype=torch.long),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = make_optimizer(
        model, lr, optimizer_name=optimizer_name, sgd_momentum=sgd_momentum
    )
    criterion = nn.CrossEntropyLoss()
    for _ in range(local_epochs):
        train_one_epoch_nb(
            model,
            loader,
            optimizer,
            criterion,
            device,
            proximal_mu=proximal_mu,
            global_params=global_param_tensors,
        )
    return get_parameters_np(model), len(dataset)


def _as_numpy_xy(X_train: ArrayOrTensor, y_train: ArrayOrTensor):
    if isinstance(X_train, torch.Tensor):
        X_np = X_train.detach().cpu().numpy()
    else:
        X_np = np.asarray(X_train)
    if isinstance(y_train, torch.Tensor):
        y_np = y_train.detach().cpu().numpy()
    else:
        y_np = np.asarray(y_train)
    return X_np, y_np


def run_fl_experiment(
    *,
    model_name: str,
    algorithm: str,
    partition: Dict[int, np.ndarray],
    alpha_label: str,
    seed: int,
    X_train: ArrayOrTensor,
    y_train: ArrayOrTensor,
    test_loader: DataLoader,
    num_features: int,
    num_classes: int,
    device: torch.device,
    cfg: Optional[dict] = None,
    mode: str = "match_notebook",
    log_every: int = 5,
    verbose: bool = True,
) -> dict:
    """
    algorithm: FedAvg | FedProx | FedTrimmedAvg | UnweightedFedAvg | WeightedFedTrimmedAvg
    mode: match_notebook | fast
    """
    cfg = dict(CFG_FULL if cfg is None else cfg)
    set_global_seed(seed)

    num_clients = cfg["num_clients"]
    num_rounds = cfg["num_rounds"]
    local_epochs = cfg["local_epochs"]
    batch_size = cfg["batch_size"]
    lr = cfg["lr"]

    if mode == "match_notebook":
        return _run_match_notebook(
            model_name=model_name,
            algorithm=algorithm,
            partition=partition,
            alpha_label=alpha_label,
            seed=seed,
            X_train=X_train,
            y_train=y_train,
            test_loader=test_loader,
            num_features=num_features,
            num_classes=num_classes,
            device=device,
            cfg=cfg,
            log_every=log_every,
            verbose=verbose,
            num_clients=num_clients,
            num_rounds=num_rounds,
            local_epochs=local_epochs,
            batch_size=batch_size,
            lr=lr,
        )
    if mode == "fast":
        return _run_fast(
            model_name=model_name,
            algorithm=algorithm,
            partition=partition,
            alpha_label=alpha_label,
            seed=seed,
            X_train=X_train,
            y_train=y_train,
            test_loader=test_loader,
            num_features=num_features,
            num_classes=num_classes,
            device=device,
            cfg=cfg,
            log_every=log_every,
            verbose=verbose,
            num_clients=num_clients,
            num_rounds=num_rounds,
            local_epochs=local_epochs,
            batch_size=batch_size,
            lr=lr,
        )
    raise ValueError(f"Unknown mode: {mode}")


def _run_match_notebook(
    *,
    model_name,
    algorithm,
    partition,
    alpha_label,
    seed,
    X_train,
    y_train,
    test_loader,
    num_features,
    num_classes,
    device,
    cfg,
    log_every,
    verbose,
    num_clients,
    num_rounds,
    local_epochs,
    batch_size,
    lr,
):
    """Bit-compatible path with run_hardware_probe / notebook."""
    X_np, y_np = _as_numpy_xy(X_train, y_train)

    # Same order as probe: seed already set by caller of set_global_seed above
    global_model = make_model(model_name, num_features, num_classes, device)
    global_params = get_parameters_np(global_model)

    loss0, acc0, f10 = evaluate_model_nb(global_model, test_loader, device)
    if verbose:
        print(
            f"  Round  0 (init) | Acc {acc0:.4f} F1 {f10:.4f} Loss {loss0:.4f}  "
            f"[{algorithm} α={alpha_label} seed={seed} mode=match_notebook]"
        )

    history = []
    for rnd in range(1, num_rounds + 1):
        client_results = []
        for cid in range(num_clients):
            idx = partition[cid]
            if len(idx) == 0:
                continue
            mu = cfg["fedprox_mu"] if algorithm == "FedProx" else 0.0
            updated, n_samp = train_client_nb(
                global_params,
                X_np[idx],
                y_np[idx],
                local_epochs=local_epochs,
                lr=lr,
                batch_size=batch_size,
                device=device,
                model_name=model_name,
                num_features=num_features,
                num_classes=num_classes,
                proximal_mu=mu,
                optimizer_name=cfg.get("optimizer", "adam"),
                sgd_momentum=float(cfg.get("sgd_momentum", 0.9)),
            )
            client_results.append((updated, n_samp))

        client_params_list = [p for p, _ in client_results]
        client_sizes = [n for _, n in client_results]

        if algorithm in ("FedAvg", "FedProx"):
            global_params = aggregate_fedavg_np(client_params_list, client_sizes)
        elif algorithm == "UnweightedFedAvg":
            global_params = aggregate_unweighted_mean_np(client_params_list)
        elif algorithm == "FedTrimmedAvg":
            global_params = aggregate_trimmed_avg_np(
                client_params_list, cfg["trimmed_fraction"]
            )
        elif algorithm == "WeightedFedTrimmedAvg":
            global_params = aggregate_weighted_trimmed_avg_np(
                client_params_list, client_sizes, cfg["trimmed_fraction"]
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        set_parameters_np(global_model, global_params)
        loss, acc, f1 = evaluate_model_nb(global_model, test_loader, device)
        history.append(dict(round=rnd, accuracy=acc, f1=f1, loss=loss))
        if verbose and (
            rnd % max(1, log_every) == 0 or rnd == 1 or rnd == num_rounds
        ):
            print(
                f"  Round {rnd:2d}/{num_rounds} | Acc {acc:.4f} F1 {f1:.4f} Loss {loss:.4f}"
            )

    accs = [h["accuracy"] for h in history]
    f1s = [h["f1"] for h in history]
    return dict(
        model=model_name,
        algorithm=algorithm,
        alpha=alpha_label,
        seed=seed,
        mode="match_notebook",
        final_acc=accs[-1],
        best_acc=max(accs),
        final_f1=f1s[-1],
        best_f1=max(f1s),
        history=history,
    )


def _run_fast(
    *,
    model_name,
    algorithm,
    partition,
    alpha_label,
    seed,
    X_train,
    y_train,
    test_loader,
    num_features,
    num_classes,
    device,
    cfg,
    log_every,
    verbose,
    num_clients,
    num_rounds,
    local_epochs,
    batch_size,
    lr,
):
    """Throughput path — do not mix with published tables."""
    if isinstance(X_train, np.ndarray):
        X_t = torch.from_numpy(np.ascontiguousarray(X_train)).float()
        y_t = torch.from_numpy(np.ascontiguousarray(y_train)).long()
        if device.type == "cuda":
            X_t = X_t.pin_memory()
            y_t = y_t.pin_memory()
    else:
        X_t, y_t = X_train, y_train

    global_model = make_model(model_name, num_features, num_classes, device)
    global_params = get_parameters_torch(global_model)
    client_model = make_model(model_name, num_features, num_classes, device)

    loss0, acc0, f10 = evaluate_model_nb(global_model, test_loader, device)
    if verbose:
        print(
            f"  Round  0 (init) | Acc {acc0:.4f} F1 {f10:.4f} Loss {loss0:.4f}  "
            f"[{algorithm} α={alpha_label} seed={seed} mode=fast]"
        )

    client_shards = []
    for cid in range(num_clients):
        idx = partition[cid]
        if len(idx) == 0:
            continue
        idx_t = torch.as_tensor(idx, dtype=torch.long)
        client_shards.append(
            (
                X_t.index_select(0, idx_t).contiguous(),
                y_t.index_select(0, idx_t).contiguous(),
            )
        )

    history = []
    for rnd in range(1, num_rounds + 1):
        client_results = []
        for X_c, y_c in client_shards:
            set_parameters_torch(client_model, global_params)
            mu = cfg["fedprox_mu"] if algorithm == "FedProx" else 0.0
            g_tensors = (
                [p.clone().detach() for p in client_model.parameters()]
                if mu > 0
                else None
            )
            n = X_c.shape[0]
            optimizer = make_optimizer(
                client_model,
                lr,
                optimizer_name=cfg.get("optimizer", "adam"),
                sgd_momentum=float(cfg.get("sgd_momentum", 0.9)),
            )
            criterion = nn.CrossEntropyLoss()
            client_model.train()
            for _ in range(local_epochs):
                perm = torch.randperm(n)
                for s in range(0, n, batch_size):
                    b = perm[s : s + batch_size]
                    xb = X_c[b].to(device, non_blocking=True)
                    yb = y_c[b].to(device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    out = client_model(xb)
                    loss = criterion(out, yb)
                    if mu > 0 and g_tensors is not None:
                        prox = sum(
                            ((lp - gp) ** 2).sum()
                            for lp, gp in zip(client_model.parameters(), g_tensors)
                        )
                        loss = loss + (mu / 2.0) * prox
                    loss.backward()
                    optimizer.step()
            client_results.append((get_parameters_torch(client_model), n))

        client_params_list = [p for p, _ in client_results]
        client_sizes = [n for _, n in client_results]
        if algorithm in ("FedAvg", "FedProx"):
            global_params = aggregate_fedavg_torch(client_params_list, client_sizes)
        elif algorithm == "UnweightedFedAvg":
            global_params = aggregate_unweighted_mean_torch(client_params_list)
        elif algorithm == "FedTrimmedAvg":
            global_params = aggregate_trimmed_avg_torch(
                client_params_list, cfg["trimmed_fraction"]
            )
        elif algorithm == "WeightedFedTrimmedAvg":
            global_params = aggregate_weighted_trimmed_avg_torch(
                client_params_list, client_sizes, cfg["trimmed_fraction"]
            )
        else:
            raise ValueError(algorithm)

        set_parameters_torch(global_model, global_params)
        loss, acc, f1 = evaluate_model_nb(global_model, test_loader, device)
        history.append(dict(round=rnd, accuracy=acc, f1=f1, loss=loss))
        if verbose and (
            rnd % max(1, log_every) == 0 or rnd == 1 or rnd == num_rounds
        ):
            print(
                f"  Round {rnd:2d}/{num_rounds} | Acc {acc:.4f} F1 {f1:.4f} Loss {loss:.4f}"
            )

    accs = [h["accuracy"] for h in history]
    f1s = [h["f1"] for h in history]
    return dict(
        model=model_name,
        algorithm=algorithm,
        alpha=alpha_label,
        seed=seed,
        mode="fast",
        final_acc=accs[-1],
        best_acc=max(accs),
        final_f1=f1s[-1],
        best_f1=max(f1s),
        history=history,
    )
