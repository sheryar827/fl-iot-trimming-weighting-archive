"""Narrow final-model capture seam for Q1-E7.

The accepted E7 protocol forbids changing the frozen engine.  During exactly one
``engine.run_fl_experiment`` call, this context manager delegates every
evaluation to the original evaluator and only retains the latest model
reference.  It consumes no RNG and performs no tensor copy until training has
returned and the original evaluator has been restored.
"""
from __future__ import annotations

from collections import OrderedDict
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from .e7_artifacts import read_checkpoint


_ACTIVE = False


class FinalModelCapture:
    def __init__(self, expected_calls: int = 31):
        self.expected_calls = int(expected_calls)
        self.calls = 0
        self.last_model = None
        self._engine = None
        self._original = None

    def __enter__(self):
        global _ACTIVE
        if _ACTIVE:
            raise RuntimeError("nested E7 final-model capture is forbidden")
        import fl_local.engine as engine

        self._engine = engine
        self._original = engine.evaluate_model_nb

        def delegated(model, loader, device):
            result = self._original(model, loader, device)
            self.calls += 1
            self.last_model = model
            return result

        delegated.__name__ = "e7_delegated_evaluate_model_nb"
        _ACTIVE = True
        engine.evaluate_model_nb = delegated
        return self

    def __exit__(self, exc_type, exc, tb):
        global _ACTIVE
        try:
            if self._engine is not None and self._original is not None:
                self._engine.evaluate_model_nb = self._original
        finally:
            _ACTIVE = False
        return False

    def export_state(self) -> OrderedDict[str, np.ndarray]:
        if self.calls != self.expected_calls:
            raise RuntimeError(
                f"captured {self.calls} evaluations; expected {self.expected_calls}"
            )
        if self.last_model is None:
            raise RuntimeError("no final global model was captured")
        return OrderedDict(
            (name, tensor.detach().cpu().numpy().copy())
            for name, tensor in self.last_model.state_dict().items()
        )


def load_checkpoint_model(
    *,
    npz_path: Path,
    spec_path: Path,
    model_name: str,
    num_features: int,
    num_classes: int,
    device,
):
    """Strictly reload the safe NPZ/JSON checkpoint into a fresh engine model."""
    import torch
    import fl_local.engine as engine

    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if (
        spec.get("model") != model_name
        or spec.get("num_features") != int(num_features)
        or spec.get("num_classes") != int(num_classes)
    ):
        raise ValueError("checkpoint model/dimension specification mismatch")
    arrays = read_checkpoint(npz_path, spec_path)
    model = engine.make_model(model_name, num_features, num_classes, device)
    expected = model.state_dict()
    if list(arrays) != list(expected):
        raise ValueError(
            "checkpoint state-key order mismatch\n"
            f"want={list(expected)}\n"
            f"got={list(arrays)}"
        )
    state = OrderedDict()
    for name, want in expected.items():
        arr = arrays[name]
        if tuple(arr.shape) != tuple(want.shape):
            raise ValueError(f"checkpoint shape mismatch for {name}")
        tensor = torch.from_numpy(np.ascontiguousarray(arr))
        if tensor.dtype != want.dtype:
            raise ValueError(
                f"checkpoint torch dtype mismatch for {name}: {tensor.dtype} != {want.dtype}"
            )
        state[name] = tensor.to(device=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def predict_loader(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """Stream ordered labels/predictions without retaining logits."""
    import torch

    n = len(loader.dataset)
    y_true = np.empty(n, dtype=np.uint8)
    y_pred = np.empty(n, dtype=np.uint8)
    pos = 0
    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            out = model(xb.to(device))
            pred = out.argmax(1).detach().cpu().numpy()
            truth = yb.detach().cpu().numpy()
            if not np.issubdtype(truth.dtype, np.integer):
                raise RuntimeError(f"prediction-loader truth is not integer: {truth.dtype}")
            if np.any((truth < 0) | (truth > 7)):
                raise RuntimeError("prediction-loader truth contains labels outside 0..7")
            if not np.issubdtype(pred.dtype, np.integer) or np.any((pred < 0) | (pred > 7)):
                raise RuntimeError("model prediction contains labels outside 0..7")
            end = pos + len(truth)
            y_true[pos:end] = truth.astype(np.uint8, copy=False)
            y_pred[pos:end] = pred.astype(np.uint8, copy=False)
            pos = end
    if pos != n:
        raise RuntimeError(f"prediction loader yielded {pos} rows, expected {n}")
    return y_true, y_pred
