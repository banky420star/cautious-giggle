import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import yaml
from loguru import logger
from sklearn.preprocessing import MinMaxScaler

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Python.agi_brain import AGIModel
from Python.perpetual_improvement import get_perpetual_improvement_system
from Python.config_utils import DEFAULT_TRADING_SYMBOLS, parse_symbol_list
from Python.data_feed import fetch_training_data
from Python.feature_pipeline import ENGINEERED_V2, ULTIMATE_150, build_lstm_feature_frame, normalize_feature_version
from alerts.telegram_alerts import TelegramAlerter

LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
LOCK_DIR = os.path.join(PROJECT_ROOT, ".tmp")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(LOCK_DIR, exist_ok=True)
logger.add(os.path.join(LOG_DIR, "lstm_training.log"), rotation="10 MB", level="INFO")


def _is_pid_alive(pid: int) -> bool:
    """Check if *pid* is alive.  Uses OpenProcess on Windows for reliability."""
    if pid <= 0:
        return False
    try:
        if sys.platform == "win32":
            import ctypes
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(0x100000, False, pid)  # SYNCHRONIZE
            if handle:
                kernel32.CloseHandle(handle)
                return True
            return False
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def _acquire_symbol_lock(symbol: str) -> bool:
    """Try to acquire a per-symbol training lock.  Returns True on success."""
    safe_name = symbol.replace("/", "_").replace("\\", "_")
    lock_path = os.path.join(LOCK_DIR, f"lstm_train_{safe_name}.lock")

    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode("utf-8"))
        os.close(fd)
        return True
    except FileExistsError:
        # Check if the holding process is still alive.
        try:
            existing_pid = int(
                open(lock_path, "r", encoding="utf-8").read().strip() or "0"
            )
        except Exception:
            existing_pid = 0

        if existing_pid and _is_pid_alive(existing_pid):
            return False  # Another trainer is genuinely running for this symbol.

        # Stale lock -- remove and retry once.
        logger.warning(
            "Removing stale LSTM lock for {} (PID {} is dead)", symbol, existing_pid
        )
        try:
            os.remove(lock_path)
        except OSError:
            return False

        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode("utf-8"))
            os.close(fd)
            return True
        except FileExistsError:
            return False


def _release_symbol_lock(symbol: str) -> None:
    """Release the per-symbol training lock if we own it."""
    safe_name = symbol.replace("/", "_").replace("\\", "_")
    lock_path = os.path.join(LOCK_DIR, f"lstm_train_{safe_name}.lock")
    try:
        if os.path.exists(lock_path):
            with open(lock_path, "r", encoding="utf-8") as fh:
                if fh.read().strip() == str(os.getpid()):
                    os.remove(lock_path)
    except Exception:
        pass


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return int(default)
    try:
        return int(str(raw).strip())
    except Exception:
        return int(default)


def _env_str(name: str, default: str) -> str:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return str(default)
    return str(raw).strip()


def _resolve_cfg_value(v):
    if isinstance(v, str) and v.startswith("ENV:"):
        return os.environ.get(v.split(":", 1)[1])
    return v


def _build_alerter():
    cfg_path = os.path.join(PROJECT_ROOT, "config.yaml")
    cfg = {}
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        except Exception:
            cfg = {}
    tel = cfg.get("telegram", {}) if isinstance(cfg, dict) else {}
    token = os.environ.get("TELEGRAM_TOKEN") or _resolve_cfg_value(tel.get("token"))
    chat_id = os.environ.get("TELEGRAM_CHAT_ID") or _resolve_cfg_value(tel.get("chat_id"))
    if not token or not chat_id:
        return TelegramAlerter(None, None)
    return TelegramAlerter(token, str(chat_id))


def compute_labels(raw_data: np.ndarray, close_col: int, atr_col: int, seq_len: int = 60):
    """Compute labels from RAW (unscaled) data using adaptive thresholds.

    This MUST be called on unscaled data so that return calculations and
    ATR-based floors use real price magnitudes.

    Labels:
        0 = HOLD   (return within noise band)
        1 = BUY    (return above positive threshold)
        2 = SELL   (return below negative threshold)

    The thresholds are set using a percentile-based approach on actual returns
    so that BUY and SELL each get ~25-30% of labels regardless of market regime.
    An ATR-scaled floor prevents labeling pure noise as a signal.

    Returns labels array aligned with sequence indices [seq_len, len(data)-1).
    """
    # --- First pass: compute all future returns to derive adaptive thresholds ---
    returns = []
    for i in range(seq_len, len(raw_data) - 1):
        prev_close = raw_data[i - 1, close_col]
        next_close = raw_data[i, close_col]
        future_ret = (next_close - prev_close) / (abs(prev_close) + 1e-8)
        returns.append(future_ret)

    if len(returns) == 0:
        return np.array([])

    returns_arr = np.array(returns)

    # Adaptive threshold: use the 30th percentile of |returns| as the noise floor,
    # ensuring at least ~30% of samples on each side become BUY/SELL.
    abs_returns = np.abs(returns_arr)
    adaptive_thr = float(np.percentile(abs_returns, 40))

    # ATR-based floor: don't label sub-noise moves as signals
    median_atr = float(np.median(np.abs(raw_data[seq_len:, atr_col])))
    median_close = float(np.median(np.abs(raw_data[seq_len:, close_col])) + 1e-8)
    atr_floor = max(0.0003, (median_atr / median_close) * 0.20)

    threshold = max(adaptive_thr, atr_floor)

    # --- Second pass: assign labels using the computed threshold ---
    labels = []
    for future_ret in returns:
        if future_ret > threshold:
            labels.append(1)   # BUY
        elif future_ret < -threshold:
            labels.append(2)   # SELL
        else:
            labels.append(0)   # HOLD

    return np.array(labels)


def create_sequences(scaled_data: np.ndarray, labels: np.ndarray, seq_len: int = 60):
    """Build (X, y) sequences from scaled features and pre-computed labels.

    Args:
        scaled_data: Feature matrix AFTER scaling (for model input).
        labels: Label array computed from RAW data via compute_labels().
                Must have length == (len(scaled_data) - seq_len - 1), aligned
                with indices [seq_len, len(scaled_data)-1).
        seq_len: Sequence length for the LSTM input windows.

    Returns:
        X: array of shape (n_samples, seq_len, n_features)
        y: array of shape (n_samples,)
    """
    if len(labels) == 0:
        return np.array([]), np.array([])

    X, y = [], []
    for i in range(len(labels)):
        seq_start = i  # labels[i] corresponds to index (seq_len + i) in data
        seq_end = seq_start + seq_len
        X.append(scaled_data[seq_start:seq_end])
        y.append(labels[i])

    return np.array(X), np.array(y)


def _train_one_symbol(
    symbol: str,
    epochs: int,
    seq_len: int,
    device: str,
    out_dir: str,
    period: str = "60d",
    interval: str = "5m",
    candles: int = 100_000,
    feature_version: str = ULTIMATE_150,
    data_source: str | None = None,
    alerter=None,
):
    # Prevent multiple concurrent trainers for the same symbol.
    if not _acquire_symbol_lock(symbol):
        logger.warning(
            "LSTM training already running for {} -- skipping duplicate", symbol
        )
        return None

    try:
        return _train_one_symbol_inner(
            symbol, epochs, seq_len, device, out_dir,
            period, interval, candles, feature_version,
            data_source, alerter,
        )
    finally:
        _release_symbol_lock(symbol)


def _train_one_symbol_inner(
    symbol: str,
    epochs: int,
    seq_len: int,
    device: str,
    out_dir: str,
    period: str = "60d",
    interval: str = "5m",
    candles: int = 100_000,
    feature_version: str = ULTIMATE_150,
    data_source: str | None = None,
    alerter=None,
):
    if alerter is not None:
        try:
            alerter.training(
                "LSTM",
                f"Start {symbol} | epochs={epochs} | seq_len={seq_len} | period={period} | tf={interval} | features={feature_version}",
            )
        except Exception:
            pass

    df = fetch_training_data(
        symbol,
        period=period,
        interval=interval,
        strict=False,
        bars=int(candles),
        min_bars=int(candles),
        source=data_source,
    )
    if df.empty or len(df) < seq_len + 50:
        logger.warning(f"insufficient data for {symbol}, skipping")
        if alerter is not None:
            try:
                alerter.alert(f"LSTM skipped {symbol}: insufficient raw data")
            except Exception:
                pass
        return None

    fdf, feature_columns = build_lstm_feature_frame(df, feature_version=feature_version)
    if len(fdf) < seq_len + 50:
        logger.warning(f"insufficient engineered rows for {symbol}, skipping")
        if alerter is not None:
            try:
                alerter.alert(f"LSTM skipped {symbol}: insufficient engineered rows")
            except Exception:
                pass
        return None

    feat = fdf[feature_columns].values.astype(np.float32)

    # Compute labels from RAW data BEFORE scaling — returns and ATR thresholds
    # must use real price magnitudes, not [0,1] normalized values.
    close_col = feature_columns.index("close") if "close" in feature_columns else 0
    atr_col = feature_columns.index("atr_14") if "atr_14" in feature_columns else close_col
    labels = compute_labels(feat, close_col, atr_col, seq_len=seq_len)

    # NOW scale features for model input
    scaler = MinMaxScaler()
    feat_scaled = scaler.fit_transform(feat)

    X, y = create_sequences(feat_scaled, labels, seq_len=seq_len)
    if len(X) == 0:
        logger.warning(f"no sequences for {symbol}")
        if alerter is not None:
            try:
                alerter.alert(f"LSTM skipped {symbol}: no sequences built")
            except Exception:
                pass
        return None

    # --- Class distribution analysis and weight computation ---
    unique, counts = np.unique(y, return_counts=True)
    class_dist = {int(k): int(v) for k, v in zip(unique, counts)}
    total_samples = int(len(y))
    class_pct = {int(k): round(100.0 * v / total_samples, 1) for k, v in zip(unique, counts)}
    logger.info(f"{symbol} | class distribution: {class_dist} | pct: {class_pct}")

    # Inverse-frequency class weights to combat imbalance
    n_classes = 3
    weight_arr = np.ones(n_classes, dtype=np.float32)
    for cls_id, cls_count in zip(unique, counts):
        weight_arr[int(cls_id)] = total_samples / (n_classes * max(1, cls_count))
    # Normalize so mean weight = 1.0
    weight_arr = weight_arr / weight_arr.mean()
    class_weights = torch.tensor(weight_arr, dtype=torch.float32).to(device)
    logger.info(f"{symbol} | class weights: HOLD={weight_arr[0]:.3f} BUY={weight_arr[1]:.3f} SELL={weight_arr[2]:.3f}")

    model = AGIModel(input_dim=len(feature_columns)).to(device)
    pis = get_perpetual_improvement_system()
    adjusted_rates = pis.adjust_learning_rate("lstm", symbol)
    adjusted_lr = adjusted_rates.get("base_lr", 1e-3)
    optimizer = torch.optim.Adam(model.parameters(), lr=adjusted_lr)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.long).to(device)

    batch_size = 64
    n_batches = max(1, len(X_tensor) // batch_size)

    model.train()
    last_loss = 0.0
    for epoch in range(epochs):
        perm = torch.randperm(len(X_tensor))
        X_epoch = X_tensor[perm]
        y_epoch = y_tensor[perm]

        correct = 0
        total = 0
        epoch_loss = 0.0

        for b in range(n_batches):
            start = b * batch_size
            end = min(start + batch_size, len(X_epoch))
            xb = X_epoch[start:end]
            yb = y_epoch[start:end]
            if len(xb) == 0:
                continue

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += float(loss.item())
            preds = logits.argmax(dim=1)
            correct += int((preds == yb).sum().item())
            total += int(yb.size(0))

        last_loss = epoch_loss / max(1, n_batches)
        acc = (correct / max(1, total)) * 100.0
        logger.info(f"{symbol} | epoch {epoch + 1}/{epochs} | loss {last_loss:.4f} | acc {acc:.2f}%")

        # Log per-class prediction distribution on last epoch to detect collapse
        if (epoch + 1) == epochs:
            model.eval()
            with torch.no_grad():
                all_logits = model(X_tensor)
                all_preds = all_logits.argmax(dim=1).cpu().numpy()
                pred_dist = {int(k): int(v) for k, v in zip(*np.unique(all_preds, return_counts=True))}
                pred_pct = {int(k): round(100.0 * v / len(all_preds), 1) for k, v in zip(*np.unique(all_preds, return_counts=True))}
                logger.info(f"{symbol} | final prediction distribution: {pred_dist} | pct: {pred_pct}")

                # Collapse detection: if any single class > 90%, warn
                max_pct = max(pred_pct.values()) if pred_pct else 0
                if max_pct > 90:
                    logger.warning(
                        f"{symbol} | COLLAPSE DETECTED: one class has {max_pct}% of predictions. "
                        f"Model may still be degenerate."
                    )
                    if alerter is not None:
                        try:
                            alerter.alert(
                                f"LSTM {symbol} COLLAPSE WARNING: {max_pct}% predictions in one class"
                            )
                        except Exception:
                            pass
            model.train()

        if alerter is not None and ((epoch + 1) == 1 or (epoch + 1) % 5 == 0 or (epoch + 1) == epochs):
            try:
                alerter.training(
                    "LSTM",
                    f"{symbol} epoch {epoch + 1}/{epochs} | loss={last_loss:.4f} | acc={acc:.2f}%",
                )
            except Exception:
                pass

    os.makedirs(out_dir, exist_ok=True)
    safe = symbol.replace("/", "_")
    model_path = os.path.join(out_dir, f"lstm_{safe}.pt")
    scaler_path = os.path.join(out_dir, f"lstm_scaler_{safe}.pkl")
    meta_path = os.path.join(out_dir, f"lstm_{safe}.meta.json")

    torch.save(model.state_dict(), model_path)

    import joblib

    joblib.dump(scaler, scaler_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "symbol": symbol,
                "feature_version": feature_version,
                "feature_columns": feature_columns,
                "seq_len": int(seq_len),
                "epochs": int(epochs),
                "samples": int(len(X)),
                "class_distribution": class_dist,
                "class_pct": class_pct,
                "class_weights": {
                    "HOLD": round(float(weight_arr[0]), 4),
                    "BUY": round(float(weight_arr[1]), 4),
                    "SELL": round(float(weight_arr[2]), 4),
                },
            },
            f,
            indent=2,
        )
    if alerter is not None:
        try:
            alerter.training("LSTM", f"Complete {symbol} | loss={last_loss:.4f} | samples={int(len(X))}")
        except Exception:
            pass

    return {
        "symbol": symbol,
        "model_path": model_path,
        "scaler_path": scaler_path,
        "meta_path": meta_path,
        "loss": last_loss,
        "samples": int(len(X)),
    }


def train_lstm(
    symbols=None,
    epochs=20,
    seq_len=60,
    period="60d",
    interval="5m",
    candles=100_000,
    feature_version: str = ULTIMATE_150,
    data_source: str | None = None,
):
    if symbols is None:
        symbols = list(DEFAULT_TRADING_SYMBOLS)

    if torch.cuda.is_available():
        device = "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    logger.info(
        f"LSTM per-symbol training on {device.upper()} | symbols={symbols} | epochs={epochs} | period={period} | tf={interval} | candles={candles:,} | features={feature_version}"
    )
    alerter = _build_alerter()
    try:
        alerter.training(
            "LSTM",
            f"Batch start | symbols={len(symbols)} | device={device.upper()} | epochs={epochs} | period={period} | tf={interval} | features={feature_version}",
        )
    except Exception:
        pass

    model_dir = os.path.join(PROJECT_ROOT, "models")
    per_symbol_dir = os.path.join(model_dir, "per_symbol")

    results = []
    for symbol in symbols:
        res = _train_one_symbol(
            symbol,
            epochs=epochs,
            seq_len=seq_len,
            device=device,
            out_dir=per_symbol_dir,
            period=period,
            interval=interval,
            candles=candles,
            feature_version=feature_version,
            data_source=data_source,
            alerter=alerter,
        )
        if res:
            results.append(res)

    if not results:
        logger.error("no symbol models trained")
        try:
            alerter.alert("LSTM batch failed: no symbol models trained")
        except Exception:
            pass
        return

    best = sorted(results, key=lambda x: x["loss"])[0]
    import shutil

    shutil.copy2(best["model_path"], os.path.join(model_dir, "lstm_agi_trained.pt"))
    shutil.copy2(best["scaler_path"], os.path.join(model_dir, "lstm_scaler.pkl"))
    shutil.copy2(best["meta_path"], os.path.join(model_dir, "lstm_agi_trained.meta.json"))
    logger.success(f"default lstm artifacts now point to best symbol model: {best['symbol']}")
    try:
        alerter.model(f"LSTM best model selected: {best['symbol']} | loss={best['loss']:.4f} | samples={best['samples']}")
    except Exception:
        pass


if __name__ == "__main__":
    cfg_path = os.path.join(PROJECT_ROOT, "config.yaml")
    symbols = None
    epochs = 20
    period = "60d"
    interval = "5m"
    candles = 100_000
    feature_version = ULTIMATE_150
    data_source = None

    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        symbols = cfg.get("trading", {}).get("symbols")
        env_symbols = os.environ.get("AGI_LSTM_SYMBOLS")
        if env_symbols:
            symbols = parse_symbol_list(env_symbols)
        tcfg = cfg.get("training", {}) or {}
        epochs = _env_int("AGI_LSTM_EPOCHS", int(tcfg.get("lstm_epochs", 20)))
        period = _env_str("AGI_LSTM_PERIOD", str(tcfg.get("lstm_period", "90d")))
        interval = _env_str("AGI_LSTM_INTERVAL", str(tcfg.get("lstm_interval", cfg.get("trading", {}).get("timeframe", "M5"))))
        candles = _env_int("AGI_LSTM_CANDLES", int(tcfg.get("lstm_candles", 100000)))
        feature_version = normalize_feature_version(
            os.environ.get("AGI_FEATURE_VERSION") or tcfg.get("feature_version", ULTIMATE_150),
            default=ULTIMATE_150,
        )
        data_source = tcfg.get("data_source")
    else:
        feature_version = normalize_feature_version(os.environ.get("AGI_FEATURE_VERSION"), default=ULTIMATE_150)
        symbols = list(DEFAULT_TRADING_SYMBOLS)

    train_lstm(
        symbols=symbols,
        epochs=epochs,
        period=period,
        interval=interval,
        candles=candles,
        feature_version=feature_version,
        data_source=data_source,
    )
