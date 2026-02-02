import numpy as np
import pandas as pd

def make_windows_two_channel(mains_1d: np.ndarray, window_size: int):
    W = int(window_size)
    half = W // 2
    n = len(mains_1d)
    if n < W:
        return None, None

    centers = np.arange(half, n - half)
    X = np.empty((len(centers), W, 2), dtype=np.float32)

    for j, c in enumerate(centers):
        w = mains_1d[c - half : c + half + 1]
        d = np.diff(w, prepend=w[0])
        X[j, :, 0] = w
        X[j, :, 1] = d

    return X, centers

def predict_regressor(regressor, X_on: np.ndarray) -> np.ndarray:
    """Supports regressor with 1 input (B,W,2) OR 2 inputs [mains(B,W,1), delta(B,W,1)]."""
    if X_on.size == 0:
        return np.array([], dtype=np.float32)

    n_inputs = len(regressor.inputs)

    if n_inputs == 1:
        return regressor.predict(X_on, verbose=0).reshape(-1).astype(np.float32)

    if n_inputs == 2:
        mains = X_on[:, :, 0:1]
        delta = X_on[:, :, 1:2]
        return regressor.predict([mains, delta], verbose=0).reshape(-1).astype(np.float32)

    raise ValueError(f"Unexpected regressor input count: {n_inputs}")

def reconstruct_from_mains(
    mains_w: np.ndarray,
    ts: np.ndarray,
    *,
    window_size: int,
    classifier,
    clf_threshold: float,
    regressor,
    regressor_output_unit: str = "kw",
    max_mains_w: float | None = None,
):
    # basic sanity filtering (optional)
    mains_w = mains_w.astype(np.float32)

    if max_mains_w is not None:
        mask = (mains_w >= 0) & (mains_w <= float(max_mains_w))
        mains_w = mains_w[mask]
        ts = ts[mask]

    X, centers = make_windows_two_channel(mains_w, window_size)
    if X is None:
        raise ValueError(f"Not enough points ({len(mains_w)}) for window_size={window_size}")

    ts_c = ts[centers]

    probs = classifier.predict(X, verbose=0).reshape(-1).astype(np.float32)
    pred_label = (probs >= float(clf_threshold)).astype(np.int32)
    on_idx = np.where(pred_label == 1)[0]

    pred_reg_w = np.zeros(len(centers), dtype=np.float32)
    if len(on_idx) > 0:
        X_on = X[on_idx].copy()

        if regressor_output_unit.lower() == "kw":
            X_on /= 1000.0
            pr_kw = predict_regressor(regressor, X_on)
            pred_reg_w[on_idx] = pr_kw * 1000.0
        else:
            pr = predict_regressor(regressor, X_on)
            pred_reg_w[on_idx] = pr

    out = pd.DataFrame({
        "timestamp": pd.to_datetime(ts_c, utc=True),
        "prob": probs,
        "pred_label": pred_label,
        "pred_regressor_w": pred_reg_w,
    }).sort_values("timestamp")

    meta = {
        "n_raw": int(len(ts)),
        "n_centers": int(len(centers)),
        "window_size": int(window_size),
        "clf_threshold": float(clf_threshold),
        "pred_on_rate": float(pred_label.mean()),
    }

    return out, meta
