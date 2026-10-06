# s38-squeeze — Hyperliquid volatility-compression scanner + convex breakout engine

Scanner for mid/low-cap HL perps: finds **Bollinger-inside-Keltner squeezes** on 1h and 4h closed bars, flags
**short-squeeze fuel** (near-zero / negative funding while price holds the top of its range), feeds a per-coin
**state engine** (`IDLE → WATCH → ARMED → FIRE → COOLDOWN`), and — with `--convex` — a **ConvexSignalEngine**
that enters only when compression turns into expansion: Donchian breakout + volume anomaly + open-interest inflow,
executed as a slippage-bounded IOC with a resting stop.

**Dry-run by default.** Without `--live` nothing is ever sent; signals, rejections and would-be orders are logged
and stored in sqlite so the whole chain can be audited and backtested before any capital is attached.

```
POST /info metaAndAssetCtxs ─► universe filter ─► closed-bar candles (1h, 4h) ─► BB(20,2) vs KC(20,1.5) squeeze
                            └► current funding + 24h fundingHistory ───────────► short-squeeze fuel
─► ScanRow per coin ─► StateEngine ─► sinks: stdout table · sqlite · Telegram (optional) · dry-run intents
                    └► ConvexSignalEngine (Donchian + vol z ≥ 2.5 + OI ≥ +3 %) ─► ConvexExecutor (IOC ± 0.08 %, stop)
```

## Run

```bash
make setup                 # python3 -m venv .venv && pip install -r requirements.txt
make test                  # 28 offline tests: indicator math, convex filters, executor vs fake gateway, fake /info server end-to-end
make once                  # one cycle, prints the table (first run loads ~2 candle series per coin)
make once ARGS=--convex    # same, plus the breakout engine in dry mode
make run ARGS=--convex     # loop every 60s; Ctrl-C to stop
.venv/bin/python3 squeeze_scanner.py --help
```

Useful flags: `--min-vol/--max-vol` (default 2e6/40e6), `--exclude BTC,ETH,SOL`, `--max-spread-bps 60`,
`--tfs 1h,4h` (the **first** TF gates the engine and the fuel range test), `--equity 5000 --risk-pct 1`
(sizing; live reads equity from the account), `--long-only` (default: both sides), `--weight 600` (API budget, see
below), `--db state/s38.db` (`--db ''` disables), `--coins AAA,BBB` (debug: restrict), `--quiet` (no table, log
lines only — use under systemd), `--convex` / `--live` / `--max-positions 3` / `--max-slippage-bps 8` (convex
engine), `--scans` / `--events N` / `--signals N` / `--orders N` (print from the db and exit; also `make scans`,
`make events`, `make signals`, `make orders`).

Optional Telegram (FIRE/ARM events, convex signals and order reports): `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID`
in `.env` (see `env.sample`).

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

## Convex signal engine (`convex_engine.py`, `--convex`)

`ConvexSignalEngine.evaluate()` runs once per scanner cycle on every eligible coin's newest **closed** 1h bar;
each (coin, bar) is judged exactly once and written to `signals` with `accepted` and the list of reasons. A
signal needs all five:

1. **Compression** — the 1h squeeze held ≥ 3 bars and ended ≤ 2 bars ago (or is still on): `bars_since_squeeze`,
   `last_squeeze_run` from `compute_squeeze`.
2. **Breakout** — bar close > max high of the *prior* 20 bars → long; < min low of the prior 20 → short. The
   breakout bar is excluded from its own channel.
3. **Volume anomaly** — `z = (v_bar − mean(v_prior20)) / σ(v_prior20) ≥ 2.5` (population σ; zero dispersion ⇒ NaN ⇒
   reject). Low-volume breakouts never pass.
4. **OI inflow** — `openInterest` is a snapshot in **coin units** (not notional, which would inflate with the price
   move itself) sampled every cycle into `scans.oi` and kept 6h in memory (reloaded from sqlite on restart):
   `OI(bar close) / OI(bar open) − 1 ≥ 3 %`, both samples within ±5 min of their timestamps, else
   `oi_history_insufficient`. The engine therefore needs ≥ 1 hour of running before its first OI confirmation.
5. **Freshness** — evaluated < 15 min after the bar close; a restart cannot fire on an old bar (`stale(...)`).

Stop = breakout-bar low (long) / high (short), at least 1 ATR(20) from the reference price. Shorts are produced
unless `--long-only`.

## Execution protocol (`ConvexExecutor`)

Per signal, in order: kill switch (`STOP` file in the working directory) → account state (live: `clearinghouseState`
equity and open positions; dry: `--equity` and simulated positions) → fresh mids (`allMids`, weight 2) → skip if
already in position, in cooldown (24 bars after a fill, 1 bar after an unfilled/failed attempt), at
`--max-positions`, or if price has already fallen back inside the Donchian range → size → send.

- **Sizing**: `size = equity × risk% / |ref − stop|`, notional capped at `min(equity × 5, 0.05 % of 24h volume)`,
  floored to `szDecimals`, HL $10 minimum (`too_small` otherwise). Leverage is set per coin to
  `min(5, maxLeverage)` (cross; isolated for `onlyIsolated` assets) before its first order.
- **Entry**: one **IOC limit** at `mid × (1 + 0.08 %)` for buys / `mid × (1 − 0.08 %)` for sells, rounded *toward*
  the mid under HL's tick rule (≤ 5 significant figures, ≤ 6 − szDecimals decimals). Hyperliquid's "market" order
  is exactly this slippage-bounded IOC. Whatever fills within the bound is kept (`filled` / `partial`); the rest is
  cancelled by the exchange and never chased (`unfilled` ⇒ 1-bar cooldown). Realized slippage vs the reference
  mid is recorded in bps.
- **Protection**: immediately after a fill, a **reduce-only stop-market** (`trigger … tpsl: sl, isMarket`) for the
  filled size at the signal stop, worst acceptable price 5 % beyond the trigger. If the stop is rejected the report
  says `POSITION UNPROTECTED` at CRITICAL level and goes to Telegram.
- **Dry mode** (default): the exact order payload, sizing and stop are logged and stored with `mode = dry`; no
  exchange request is made and the SDK is not even imported.
- **Live mode**: `--live` plus `HL_API_WALLET_KEY` and `HL_ACCOUNT_ADDRESS` (optionally `HL_SUBACCOUNT_ADDRESS`)
  in the environment, otherwise the process refuses to start. Use an API wallet (agent key, trading-only) for a
  dedicated $1k sub-account, never the account's private key. Startup logs the signer and the address being traded
  (both public); the key is never logged or stored. Orders go through the official `hyperliquid-python-sdk`
  (`Exchange.order`, `Exchange.update_leverage`) in a worker thread; exchange requests are weight 1.

There is no exit management beyond the initial stop yet — that is a later milestone (or hl-trailstop).

## Output

Table legend is printed under each table. sqlite tables (`state/s38.db`): `scans` (one row per coin per cycle:
price, funding, OI in coin units, volume, spread, both TFs' squeeze metrics, fuel, state, score), `events` (every
state transition, FIRE rows carry the intent JSON), `funding_hourly` (realized hourly prints), `universe`
(eligibility counts), `signals` (one row per evaluated breakout bar with every filter value, `accepted`, `reasons`),
`orders` (every execution report: plan, fill, slippage, stop, raw exchange response). Times are UTC ms.

## API budget — read before running on a shared IP

Every info call used here (`metaAndAssetCtxs`, `candleSnapshot`, `fundingHistory`) has weight 20 against HL's
**1200/min per IP** (`allMids`, `clearinghouseState`: 2; exchange orders: 1). The limit is per IP, so every process
on the same box or behind the same router shares it: the house IP carries the live `hl-trailstop` bot (2026-09-13
429 incident), hospitality1 carries the live s16 book — on the first run there (2026-10-06) the **very first**
request was answered 429, i.e. the IP was already saturated before this scanner sent anything.

`--weight` is therefore a *ceiling*, not a reservation. The limiter has no start-up burst (two calls), halves its
rate on every 429 (floor 10 % of the ceiling) and owes 30 s of silence, then recovers linearly over 10 minutes
without further 429s; each cycle line logs `N×429, budget X/min`. Defaults: 600 from the CLI, 300 in the
hospitality1 unit. Cold start = 2 candle series per eligible coin: 48 coins ⇒ 96 calls ⇒ ~6.5 min at 300/min
(observed: 5.5 min at 400 with 17 retried 429s); afterwards a cycle costs 1 call plus one refresh per coin per
closed bar (1h: ~48 calls once an hour; 4h: every 4 hours). If s16's journal shows 429s in the minutes after this
scanner starts, lower the ceiling further or move the scanner to a box with its own IP. Only a box with no other HL
consumer should run `--weight 1000`.

## Deploy on hospitality1

Same layout as s35-pullback: code in `/home/dev/s38-squeeze`, runs as root under systemd with system `python3`
(≥ 3.10), state in `state/s38.db`, optional Telegram via `.env`.

```bash
ssh hospitality1
sudo -i
git clone https://github.com/jianchensf/s38-squeeze /home/dev/s38-squeeze     # later: cd … && git pull
bash /home/dev/s38-squeeze/deploy/install_on_box.sh                           # deps, .env, tests, unit file
cd /home/dev/s38-squeeze && python3 squeeze_scanner.py --once --convex --weight 300   # read-only + dry orders
systemctl enable --now s38-squeeze && journalctl -u s38-squeeze -f            # 1 log line per cycle + events
python3 squeeze_scanner.py --scans      # latest table from the db      make scans
python3 squeeze_scanner.py --events 30  # last 30 state transitions     make events
python3 squeeze_scanner.py --signals    # convex signals + rejections   make signals
python3 squeeze_scanner.py --orders     # order reports (dry or live)   make orders
```

Upgrade: `git pull && bash deploy/install_on_box.sh && systemctl restart s38-squeeze` (the sqlite schema migrates
itself; a v1 database gains the `oi` column and the `signals` / `orders` tables on first start).

## Known limits

- HIP-3 builder dexes are not scanned (main perp universe only).
- `funding` from `metaAndAssetCtxs` is the current hour's rate and moves within the hour; the realized prints
  are in `funding_hourly`.
- Signal latency = the cycle that follows the hourly close: 1–3 min at 600 weight/min, up to ~7 min at 400 when
  the 1h and 4h refresh coincide. Entries are therefore minutes after the close, never intra-bar.
- HL has no OI history endpoint; OI confirmation only works from the scanner's own samples (≥ 1 h of uptime).
- The convex engine's stop is the only exit. No trailing, no take-profit, no time stop yet.
- Nothing here is a backtest. The squeeze → expansion thesis, the volume/OI filters and the stop placement are all
  unvalidated on HL mid-caps until the study below is done.

## Graduation

1. Run `--convex` in dry mode on hospitality1 and let `signals` / `orders` accumulate. Review with `make signals`
   and `make orders`: how many breakouts pass each filter, where the dry fills would have been, and what happened
   after (the `scans` table has mark prices every minute for the forward path).
2. **Historical study** (proposed, not started): squeeze releases and Donchian/volume/OI breakouts on the current
   universe from `candleSnapshot` (up to 5000 1h bars ≈ 200 days per coin): forward |return| and signed return at
   4/12/24/48 bars, split by direction, by fuel / no fuel and by 4h confirmation; n, mean, median, SE, skew, hit rate
   per leg against unconditional bars of the same coins. OI cannot be backtested (no history endpoint), so the OI
   filter's marginal value can only be measured live from the dry log.
3. Only if 1–2 hold up: `--live` on a dedicated $1k sub-account driven by an API wallet, `--max-positions 3`,
   1 % risk, with exit management added first.
