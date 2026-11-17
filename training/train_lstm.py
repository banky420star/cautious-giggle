import json
import time
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from loguru import logger
from sklearn.preprocessing import MinMaxScaler

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Python.agi_brain import AGIModel
from Python.config_utils import DEFAULT_TRADING_SYMBOLS, parse_symbol_list
from Python.data_feed import fetch_training_data
from Python.feature_pipeline import ENGINEERED_V2, ULTIMATE_150, build_lstm_feature_frame, normalize_feature_version
from alerts.telegram_alerts import TelegramAlerter

LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
logger.add(os.path.join(LOG_DIR, "lstm_training.log"), rotation="10 MB", level="INFO")


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


def create_sequences(
    data: np.ndarray,
    close_col: int,
    atr_col: int,
    rsi_col: int,
    seq_len: int = 60,
    label_data: np.ndarray | None = None,
    labels_only: bool = False,
):
    """X = scaled feature windows; y computed from *raw* (unscaled) prices.

    Labels used to be computed on MinMax-scaled columns, which made the return and ATR
    thresholds meaningless. Pass the unscaled matrix as label_data.
    """
    raw = data if label_data is None else label_data
    X, y = [], []
    for i in range(seq_len, len(data) - 1):
        if not labels_only:
            X.append(data[i - seq_len : i])

        prev_close = raw[i - 1, close_col]
        next_close = raw[i, close_col]
        future_ret = (next_close - prev_close) / (abs(prev_close) + 1e-8)

        # ATR known at the last bar of the window (no lookahead in the threshold).
        atr_norm = abs(raw[i - 1, atr_col] / (abs(prev_close) + 1e-8))
        rsi = raw[i - 1, rsi_col]

        # Volatility-relative threshold with a floor around round-trip cost (~1.5 bps).
        up_thr = max(0.00015, atr_norm * 0.35)
        dn_thr = -up_thr

        if future_ret > up_thr and rsi > 52:
            y.append(1)
        elif future_ret < dn_thr and rsi < 48:
            y.append(2)
        else:
            y.append(0)

    return np.array(X), np.array(y)


def _raw_label_matrix(raw_df: pd.DataFrame, fdf: pd.DataFrame) -> np.ndarray:
    """[close, ATR14, RSI14(0-100)] from raw OHLC, row-aligned with the feature frame.

    The ultimate_150 feature set has no raw close/atr_14/rsi_14 columns, so labels must
    come from the price data itself (previously they silently used feature column 0).
    """
    src = raw_df.copy()
    if "time" in src.columns:
        src["time"] = pd.to_datetime(src["time"], utc=True, errors="coerce")
        src = src.set_index("time")
    close = pd.to_numeric(src["close"], errors="coerce")
    high = pd.to_numeric(src["high"], errors="coerce")
    low = pd.to_numeric(src["low"], errors="coerce")
    prev = close.shift(1).fillna(close)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14, min_periods=1).mean()
    delta = close.diff().fillna(0.0)
    gain = delta.clip(lower=0).rolling(14, min_periods=1).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean()
    rsi = (100.0 - 100.0 / (1.0 + gain / (loss + 1e-12))).fillna(50.0)
    lab = pd.DataFrame({"close": close, "atr": atr, "rsi": rsi})
    if len(lab) == len(fdf):
        return lab.to_numpy(dtype=np.float64)
    idx = pd.to_datetime(fdf.index, utc=True, errors="coerce") if isinstance(fdf.index, pd.DatetimeIndex) else fdf.index
    return lab.reindex(idx).ffill().to_numpy(dtype=np.float64)


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
    # Chronological split: the scaler is fit on the training rows only, and the last
    # val_frac of the data is used to select the checkpoint.
    val_frac = float(os.environ.get("AGI_LSTM_VAL_FRAC", "0.2"))
    n_train_rows = int(len(feat) * (1.0 - val_frac))
    scaler = MinMaxScaler()
    scaler.fit(feat[:n_train_rows])
    feat_scaled = scaler.transform(feat)

    label_raw = _raw_label_matrix(df, fdf)
    # Labels only; windows are sliced on the fly from one copy of the feature matrix
    # (materialising every overlapping window used ~3.6 GB and exhausted MPS memory).
    _, y_all = create_sequences(feat_scaled, 0, 1, 2, seq_len=seq_len, label_data=label_raw, labels_only=True)
    n_seq = len(y_all)
    # Sequence k = rows [k, k+seq_len); its label is bar k+seq_len. Train if that bar is a train row.
    split_idx = max(1, min(n_seq - 1, n_train_rows - seq_len))
    y, y_val = y_all[:split_idx], y_all[split_idx:]
    if n_seq <= 1 or len(y) == 0:
        logger.warning(f"no sequences for {symbol}")
        if alerter is not None:
            try:
                alerter.alert(f"LSTM skipped {symbol}: no sequences built")
            except Exception:
                pass
        return None

    model = AGIModel(input_dim=len(feature_columns)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    feat_t = torch.tensor(feat_scaled, dtype=torch.float32, device=device)
    offsets = torch.arange(seq_len, device=device)
    y_all_t = torch.tensor(y_all, dtype=torch.long, device=device)
    train_idx = torch.arange(0, split_idx, device=device)
    val_idx = torch.arange(split_idx, n_seq, device=device)

    def batch(idx):
        # Gather only the rows this batch needs: (B, seq_len, F). Indexing an unfold() view
        # made MPS materialise every window (~3.6 GB per call).
        return feat_t[idx[:, None] + offsets[None, :]], y_all_t[idx]

    # Class-weighted loss: HOLD dominates, and an unweighted model learns to always say HOLD.
    counts = np.bincount(y, minlength=3).astype(np.float64)
    weights = torch.tensor(counts.sum() / (3.0 * np.maximum(counts, 1.0)), dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    best_val_loss = float("inf")
    best_state = None
    val_acc = 0.0

    batch_size = int(os.environ.get("AGI_LSTM_BATCH", "512"))
    eval_batch = 2048
    patience = int(os.environ.get("AGI_LSTM_PATIENCE", "4"))
    stale = 0
    n_batches = max(1, (len(train_idx) + batch_size - 1) // batch_size)

    from Python.train_progress import ProgressWriter

    prog = ProgressWriter("lstm", symbol)
    vc = np.bincount(y_val, minlength=3) if len(y_val) else np.zeros(3)
    prog.update(force=True, epochs=epochs, epoch=0, batch=0, n_batches=n_batches, train_samples=int(len(train_idx)),
                val_samples=int(len(val_idx)), patience=patience,
                majority_val_acc=float(vc.max() / max(1, vc.sum()) * 100.0),
                val_class_pct={"hold": float(vc[0] / max(1, vc.sum()) * 100), "up": float(vc[1] / max(1, vc.sum()) * 100),
                               "down": float(vc[2] / max(1, vc.sum()) * 100)})
    t_train = time.time()
    model.train()
    last_loss = 0.0
    for epoch in range(epochs):
        t_epoch = time.time()
        perm = train_idx[torch.randperm(len(train_idx), device=device)]
        correct = 0
        total = 0
        epoch_loss = 0.0

        for b in range(n_batches):
            idx = perm[b * batch_size:(b + 1) * batch_size]
            if len(idx) == 0:
                continue
            xb, yb = batch(idx)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            lv = float(loss.item())
            epoch_loss += lv
            correct += int((logits.argmax(dim=1) == yb).sum().item())
            total += int(yb.size(0))
            el = max(1e-6, time.time() - t_epoch)
            prog.update(recent=lv, epoch=epoch + 1, batch=b + 1, run_loss=epoch_loss / (b + 1),
                        run_acc=correct / max(1, total) * 100.0, samples_per_s=total / el,
                        epoch_eta_s=(n_batches - b - 1) * el / (b + 1), phase="train")

        last_loss = epoch_loss / max(1, n_batches)
        acc = (correct / max(1, total)) * 100.0
        prog.update(force=True, phase="validating")
        model.eval()
        with torch.no_grad():
            if len(val_idx):
                tot_loss, tot_ok = 0.0, 0
                for vs in range(0, len(val_idx), eval_batch):
                    xv, yv = batch(val_idx[vs:vs + eval_batch])
                    vl = model(xv)
                    tot_loss += float(criterion(vl, yv).item()) * len(yv)
                    tot_ok += int((vl.argmax(dim=1) == yv).sum().item())
                v_loss = tot_loss / len(val_idx)
                v_acc = tot_ok / len(val_idx) * 100.0
            else:
                v_loss, v_acc = last_loss, acc
        model.train()
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            val_acc = v_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        logger.info(
            f"{symbol} | epoch {epoch + 1}/{epochs} | loss {last_loss:.4f} | acc {acc:.2f}% | val_loss {v_loss:.4f} | val_acc {v_acc:.2f}%"
        )
        if alerter is not None and ((epoch + 1) == 1 or (epoch + 1) % 5 == 0 or (epoch + 1) == epochs):
            try:
                alerter.training(
                    "LSTM",
                    f"{symbol} epoch {epoch + 1}/{epochs} | loss={last_loss:.4f} | acc={acc:.2f}%",
                )
            except Exception:
                pass
        prog.epoch_done(epoch=epoch + 1, loss=last_loss, acc=acc, val_loss=v_loss, val_acc=v_acc,
                        best=bool(stale == 0), secs=time.time() - t_epoch)
        if stale >= patience:
            logger.info(f"{symbol} | early stop: no val improvement for {patience} epochs")
            break

    prog.finish(phase="done", best_val_loss=best_val_loss, best_val_acc=val_acc, total_s=time.time() - t_train)
    os.makedirs(out_dir, exist_ok=True)
    safe = symbol.replace("/", "_")
    model_path = os.path.join(out_dir, f"lstm_{safe}.pt")
    scaler_path = os.path.join(out_dir, f"lstm_scaler_{safe}.pkl")
    meta_path = os.path.join(out_dir, f"lstm_{safe}.meta.json")

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), model_path)
    majority_val_acc = float(np.bincount(y_val, minlength=3).max() / max(1, len(y_val)) * 100.0) if len(y_val) else 0.0

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
                "samples": int(len(y)),
                "val_samples": int(len(y_val)),
                "val_loss": float(best_val_loss),
                "val_acc": float(val_acc),
                "val_majority_acc": majority_val_acc,
                "scaler_fit": "train_rows_only",
                "labels": "raw_prices_v2",
            },
            f,
            indent=2,
        )
    if alerter is not None:
        try:
            alerter.training("LSTM", f"Complete {symbol} | loss={last_loss:.4f} | samples={int(len(y))}")
        except Exception:
            pass

    return {
        "symbol": symbol,
        "model_path": model_path,
        "scaler_path": scaler_path,
        "meta_path": meta_path,
        "loss": float(best_val_loss),
        "train_loss": last_loss,
        "samples": int(len(y)),
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
