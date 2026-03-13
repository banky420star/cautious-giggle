import json

from alerts.telegram_alerts import TelegramAlerter


def test_telegram_alerter_loads_legacy_card_ids(tmp_path):
    path = tmp_path / "telegram_cards.json"
    path.write_text(json.dumps({"heartbeat": 17}), encoding="utf-8")

    alerter = TelegramAlerter(None, None, cards_state_path=str(path))
    summary = alerter.state_summary(limit=5)

    assert summary["card_count"] == 1
    assert summary["cards"][0]["message_id"] == 17
    assert summary["cards"][0]["delivery_status"] == "dashboard_only"


def test_telegram_alerter_persists_modern_metadata(tmp_path):
    path = tmp_path / "telegram_cards.json"
    alerter = TelegramAlerter("token", "chat", cards_state_path=str(path))

    def fake_api(method, payload):
        if method == "sendMessage":
            return {"message_id": 51}
        return None

    alerter._api = fake_api
    alerter.alert("Operator action completed")

    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted["alerts"]["message_id"] == 51
    assert persisted["alerts"]["delivery_status"] == "both"
    assert persisted["alerts"]["category"] == "incident"


def test_telegram_alerter_persists_training_cycle_card(tmp_path):
    path = tmp_path / "telegram_cards.json"
    alerter = TelegramAlerter("token", "chat", cards_state_path=str(path))

    def fake_api(method, payload):
        if method == "sendMessage":
            return {"message_id": 88}
        return None

    alerter._api = fake_api
    alerter.training_cycle(
        "complete",
        ["BTCUSDm", "XAUUSDm"],
        {
            "mode": "per_symbol",
            "skip_lstm": True,
            "skip_dreamer": True,
            "symbols": [
                {"symbol": "BTCUSDm", "wins": True, "passes_thresholds": True},
                {"symbol": "XAUUSDm", "wins": False, "passes_thresholds": False},
            ],
        },
        detail="promoted=BTCUSDm",
    )

    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted["training_cycle"]["message_id"] == 88
    assert persisted["training_cycle"]["delivery_status"] == "both"
    assert persisted["training_cycle"]["category"] == "training"


def test_telegram_alerter_records_retry_after_on_rate_limit(tmp_path):
    path = tmp_path / "telegram_cards.json"
    alerter = TelegramAlerter("token", "chat", cards_state_path=str(path))

    def fake_api(method, payload):
        alerter._last_api_error = {"status_code": 429, "retry_after": 60, "body": {"ok": False}}
        return None

    alerter._api = fake_api
    sent = alerter.training_cycle(
        "running",
        ["BTCUSDm"],
        {"mode": "per_symbol", "symbols": []},
        detail="rate-limit probe",
    )

    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert sent is None or sent is False
    assert persisted["training_cycle"]["delivery_status"] == "dashboard_only"
    assert persisted["training_cycle"]["next_retry_utc"] is not None
    assert persisted["training_cycle"]["last_text"]
