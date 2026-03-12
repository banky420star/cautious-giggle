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
