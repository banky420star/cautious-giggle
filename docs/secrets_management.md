# Secrets Management

## Required Secrets

| Secret | Purpose | Where Set | Used By |
|--------|---------|-----------|---------|
| `MT5_LOGIN` | MetaTrader 5 account number | Env var or `.env` | Server_AGI |
| `MT5_PASSWORD` | MetaTrader 5 account password | Env var or `.env` | Server_AGI |
| `MT5_SERVER` | MetaTrader 5 broker server | Env var or `.env` | Server_AGI |
| `TELEGRAM_TOKEN` | Telegram bot API token | Env var or `.env` | Alerter |
| `TELEGRAM_CHAT_ID` | Telegram chat/group ID | Env var or `.env` | Alerter |
| `AGI_TOKEN` | API token for the live server | Env var or `.env` | start scripts / compose |
| `N8N_PASSWORD` | n8n dashboard password | Env var (docker-compose) | n8n |

## Config File Reference

Secrets in `config.yaml` should use the `ENV:VARIABLE_NAME` pattern to reference environment variables:

```yaml
mt5:
  login: ENV:MT5_LOGIN
  password: ENV:MT5_PASSWORD
  server: ENV:MT5_SERVER
telegram:
  token: ENV:TELEGRAM_TOKEN
  chat_id: ENV:TELEGRAM_CHAT_ID
```

Startup scripts load `.env` automatically when present. Copy `.env.example` to `.env`, fill the values, and keep `.env` out of source control.

Live mode validation is strict:

- MT5 credentials must resolve to non-empty values.
- Telegram credentials must be either fully configured or fully omitted.
- Placeholder values such as `YOUR_BOT_TOKEN_HERE` are rejected in live mode.
- `ENV:` references are trimmed and validated before startup.

## Rotation Procedures

### MT5 Credentials
1. Update password in your broker's web portal
2. Update `MT5_PASSWORD` in your environment or `.env`
3. Restart Server_AGI

### Telegram Bot Token
1. Create new token via @BotFather on Telegram
2. Update `TELEGRAM_TOKEN` in your environment or `.env`
3. Restart Server_AGI (no downtime required - alerts are non-critical)

### n8n Dashboard
1. Set new `N8N_PASSWORD` env var
2. Run `docker-compose up -d n8n` to restart only n8n

## What Happens If Absent

| Secret | Behavior |
|--------|----------|
| MT5 credentials | **Live mode blocked** - RuntimeError at startup |
| Telegram credentials | Alerts disabled if omitted; partial live configuration is blocked |
| N8N_PASSWORD | docker-compose fails to start n8n service |
