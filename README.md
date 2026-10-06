# s38-squeeze — Hyperliquid volatility-compression scanner

Read-only scanner for mid/low-cap HL perps: finds **Bollinger-inside-Keltner squeezes** on 1h and 4h closed bars,
flags **short-squeeze fuel** (near-zero / negative funding while price holds the top of its range), and feeds a
per-coin **execution state engine** (`IDLE → WATCH → ARMED → FIRE → COOLDOWN`) that emits risk-sized intents.

**It never places orders.** FIRE events carry an `ExecutionIntent` (side, size, stop, notional) for a separate
executor; the one wired in here only logs. Every cycle is written to sqlite so the signals can be audited and
backtested before any capital is attached.

```
POST /info metaAndAssetCtxs ─► universe filter ─► closed-bar candles (1h, 4h) ─► BB(20,2) vs KC(20,1.5) squeeze
                            └► current funding + 24h fundingHistory ───────────► short-squeeze fuel
─► ScanRow per coin ─► StateEngine ─► sinks: stdout table · sqlite · Telegram (optional) · dry-run intents
```

## Run

```bash
make setup                 # python3 -m venv .venv && pip install -r requirements.txt
make test                  # 10 offline tests: indicator math + fake /info server end-to-end
make once                  # one cycle, prints the table (first run loads ~2 candle series per coin)
make run                   # loop every 60s; Ctrl-C to stop
.venv/bin/python3 squeeze_scanner.py --help
```

Useful flags: `--min-vol/--max-vol` (default 2e6/40e6), `--exclude BTC,ETH,SOL`, `--max-spread-bps 60`,
`--tfs 1h,4h` (the **first** TF gates the engine and the fuel range test), `--equity 5000 --risk-pct 1`
(intent sizing), `--allow-shorts`, `--weight 600` (API budget, see below), `--db state/s38.db` (`--db ''` disables),
`--coins AAA,BBB` (debug: restrict), `--quiet` (no table, log lines only — use under systemd),
`--scans` / `--events N` (print from the db and exit; `make scans`, `make events`).

Optional Telegram (FIRE + ARM events): put `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` in `.env` (see `env.sample`).

## Definitions

**Universe** (`filter_universe`): drop `--exclude` names (BTC, ETH, SOL), delisted, no mid price, 24h notional
volume outside [$2M, $40M], impact-price spread > 60 bps (slippage proxy for small clips). Exclusion counts are
logged every cycle and stored in `universe`.

**Squeeze** (`compute_squeeze`, closed bars only, UTC-aligned grid, the open bar is always dropped):
- Bollinger: `SMA(20) ± 2.0·σ` (population σ, as TradingView). Keltner: `EMA(20) ± 1.5·SMA(TrueRange, 20)`.
- `squeeze_on` = upper BB < upper KC **and** lower BB > lower KC (fully inside).
- `squeeze_ratio` = BB width / KC width (< 1 inside). `bbw_pctile` = share of the last 120 bars whose relative BB
  width ≤ today's (0 = tightest). `squeeze_bars` = consecutive closed bars in squeeze.
- `released` = on at bar[-2], off at bar[-1]; `release_dir` = sign of the TTM-style momentum proxy
  (close − midpoint of Donchian-mid and SMA).
- `range_pos` = (close − 20-bar low) / (20-bar high − 20-bar low).

**Fuel** (`compute_fuel`): HL funding is hourly; the neutral rate is 0.01%/8h = 1.25e-5/h. `near_zero` = funding
< 0.5 × neutral (negative always qualifies). `short_squeeze_fuel` = near_zero **and** 1h `range_pos ≥ 0.75`.
For coins in a squeeze the last 24 realized hourly prints are pulled (`fundingHistory`, once per coin per hour):
`neg_hours_24h ≥ 6` ⇒ `sustained_negative` (shown as `*`). Table column `F8h%` is funding × 8 (neutral = +0.0100).

**State engine** (`StateEngine`, evaluated every cycle on the gating TF = first of `--tfs`):
- `WATCH`: squeeze on in any TF. `ARMED`: 1h squeeze on **and** (fuel **or** 4h squeeze also on).
- `FIRE`: the 1h squeeze released on the newest closed bar after ≥ 3 bars in squeeze (fires even from `IDLE`, so a
  restart cannot miss a release; `events` dedupes per bar). Emits an intent, then `COOLDOWN` for 12 bars.
- `DROP`: coin left the universe for > 1h.

**Intent** (`build_intent`): stop = Keltner basis (channel mean; 1 ATR fallback), size = `equity × risk% / stop
distance`, notional capped at `min(equity × 5, 0.05% of 24h volume)`, floored to `szDecimals`, HL $10 minimum.
Long only unless `--allow-shorts`.

**Score** is a display heuristic (squeeze on both TFs, tightness percentile, bars in squeeze, fuel). It is not a
validated predictor — see "Next".

## Output

Table legend is printed under each table. sqlite tables (`state/s38.db`): `scans` (one row per coin per cycle:
price, funding, OI, volume, spread, both TFs' squeeze metrics, fuel, state, score), `events` (every transition,
FIRE rows carry the intent JSON), `funding_hourly` (realized hourly prints), `universe` (eligibility counts).
Times are UTC ms.

## API budget — read before running on a shared IP

Every call used here (`metaAndAssetCtxs`, `candleSnapshot`, `fundingHistory`) has weight 20 against HL's
**1200/min per IP**. The limit is per IP, so every process on the same box or behind the same router shares it:
the house IP carries the live `hl-trailstop` bot (2026-09-13 429 incident), hospitality1 carries the live s16
book. Defaults: `--weight 600` from the CLI, `--weight 400` in the hospitality1 unit. Cold start = 2 candle series
per eligible coin: ~70 coins ⇒ 140 calls ⇒ ~5 min at 600/min, ~7 min at 400/min; afterwards a cycle costs 1 call
plus one refresh per coin per closed bar (1h: ~70 calls once an hour; 4h: every 4 hours). Only a box with no other
HL consumer should run `--weight 1000`. Steady-state load is small enough that a WebSocket candle feed is not
worth the complexity yet.

## Deploy on hospitality1

Same layout as s35-pullback: code in `/home/dev/s38-squeeze`, runs as root under systemd with system `python3`
(≥ 3.10), state in `state/s38.db`, optional Telegram via `.env`.

```bash
ssh hospitality1
sudo -i
git clone https://github.com/jianchensf/s38-squeeze /home/dev/s38-squeeze     # later: cd … && git pull
bash /home/dev/s38-squeeze/deploy/install_on_box.sh                           # deps, .env, tests, unit file
cd /home/dev/s38-squeeze && python3 squeeze_scanner.py --once --weight 400    # read-only check, ~7 min cold start
systemctl enable --now s38-squeeze && journalctl -u s38-squeeze -f            # 1 log line per cycle + events
python3 squeeze_scanner.py --scans      # latest table from the db      make scans
python3 squeeze_scanner.py --events 30  # last 30 state transitions     make events
```

Upgrade: `git pull && bash deploy/install_on_box.sh && systemctl restart s38-squeeze`.

## Known limits

- HIP-3 builder dexes are not scanned (main perp universe only).
- `funding` from `metaAndAssetCtxs` is the current hour's rate and moves within the hour; the realized prints
  are in `funding_hourly`.
- Signal timing: a release is seen 1–2 minutes after the hourly close (cycle + refresh gap).
- Nothing here is a backtest. The squeeze → expansion thesis, the fuel filter and the stop placement are all
  unvalidated on HL mid-caps until the study below is done.

## Next (proposed, not started)

1. **Historical study** of squeeze releases on the current universe using `candleSnapshot` (up to 5000 1h bars
   ≈ 200 days per coin): forward |return| and signed return at 4/12/24/48 bars after release, split by direction,
   by fuel/no-fuel and by 4h-confirmation; report n, mean, median, SE, skew, hit rate per leg, and compare against
   unconditional bars of the same coins. Decide from that whether an entry rule deserves paper.
2. Only then: executor sink (maker entry at release-bar close, KC-basis stop, trail), paper mode first.
