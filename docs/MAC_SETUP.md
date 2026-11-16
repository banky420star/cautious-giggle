# Running on macOS (Exness Public Trader API)

The `MetaTrader5` Python package works only on Windows. On a Mac the bot talks to Exness
directly over HTTPS through `Python/exness_api.py`. That module gives the bot an
MT5-compatible interface, so the executor, risk engine and data code run unchanged.

## 1. Exness side
1. Open an **API-enabled Exness trading account**. Standard MetaTrader accounts cannot
   connect to the API.
   - Use a separate account just for the bot. The API has no magic numbers, so every
     position on the account counts as the bot's.
   - Start on demo if one is offered.
2. Complete full verification (proof of identity and proof of address).
3. In **Personal Area → Key Management**, create an API key:
   - Pick the bot's trading account.
   - Give it trading permissions.
   - Set an expiry date, and restrict it to your IP if you can.
   - Let Exness generate the key pair.
4. Save what Key Management gives you:
   - the **key ID** (`exnsk-...`)
   - the **private key** (128 hex characters, shown once)
   - the **Gateway host**, from the DNS widget
   ```bash
   mkdir -p ~/.exness && chmod 700 ~/.exness
   pbpaste > ~/.exness/api_private_key   # with the private key on the clipboard
   chmod 600 ~/.exness/api_private_key
   ```

## 2. Install
```bash
cd /Volumes/AI_DRIVE/cautious-giggle
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt      # MetaTrader5 is skipped automatically on macOS
cp config.yaml.example config.yaml   # broker.backend: exness_api is already set
```
Put the credentials in your shell profile, not in `config.yaml`:
```bash
export EXNESS_ACCOUNT_ID=...
export EXNESS_API_KEY=...
export EXNESS_PRIVATE_KEY_FILE=~/.exness/api_private_key
export EXNESS_BASE_URL=https://<gateway host from Key Management>
```

## 3. Check the connection (read-only, places no orders)
```bash
python -m Python.exness_api check BTCUSDm XAUUSDm
```
The check prints:
- the account
- API limits
- instruments, with the `BTCUSDm → BTCUSD` mapping
- volume steps
- a live bid/ask quote
- M5 history depth
- open positions
- recent deals

**Do not go live until every line looks right.**

## 4. Train (no account needed)
Training uses broker history when it is available and falls back to Dukascopy, which
works without any credentials. Apple-silicon GPU (MPS) is used automatically for the LSTM.
```bash
python tools/champion_cycle.py
```
A model is promoted only if it passes the out-of-sample gates in `config.yaml`
(`evaluation:`).

## 5. Run live
```bash
AGI_BROKER_BACKEND=exness_api python Python/Server_AGI.py
```
- Keep the Mac awake while it runs: `caffeinate -dimsu python Python/Server_AGI.py`.
  BTC trades 24/7, and a sleeping laptop leaves positions unmanaged. The broker-side SL/TP
  still protect each position.
- For unattended running, a small always-on Linux VPS is more reliable than a laptop.

## Notes and limits
- **Quotes** come from the latest bid/ask M1 candle over REST, refreshed every second
  when needed. Decisions are made once per M5 bar, so this is sufficient. WebSocket
  ticks are not used yet.
- **Orders are asynchronous.** The adapter waits for each operation to be confirmed
  (up to 8 s) and maps Exness error codes to MT5 retcodes, so the executor's retry and
  halt rules still apply.
- **API reference:** the official Exness API Developer Portal, https://www.exness-api.com.
  The client was checked against its signing and Trader API pages. No demo sandbox is
  documented, so run `check` (read-only) first, then trade the minimum volume.
