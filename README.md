# s38-squeeze — Hyperliquid volatility-compression scanner + convex breakout engine

Scanner for mid/low-cap HL perps: finds **Bollinger-inside-Keltner squeezes** on 1h and 4h closed bars, flags
**short-squeeze fuel** (near-zero / negative funding while price holds the top of its range), feeds a per-coin
**state engine** (`IDLE → WATCH → ARMED → FIRE → COOLDOWN`), and — with `--convex` — a **ConvexSignalEngine**
that enters only when compression turns into expansion: Donchian breakout + volume anomaly + open-interest inflow,
executed as a slippage-bounded IOC with a resting stop.

**Dry-run by default.** Without `--live` nothing is ever sent; signals, rejections, paper fills and paper exits
are logged and stored in sqlite so the whole chain — entry filters, sizing, stops, breaker — can be audited on
the realized paper record before any capital is attached.

```
POST /info metaAndAssetCtxs ─► universe filter ─► closed-bar candles (1h, 4h) ─► BB(20,2) vs KC(20,1.5) squeeze
                            └► current funding + 24h fundingHistory ───────────► short-squeeze fuel
─► ScanRow per coin ─► StateEngine ─► sinks: stdout table · sqlite · Telegram (optional) · dry-run intents
                    └► ConvexSignalEngine (Donchian + vol z ≥ 2.5 + OI ≥ +3 %) ─► ConvexExecutor (IOC ± 0.08 %)
                       ConvexRiskManager: ¼-Kelly ≤ 4 % · 1.5×ATR14 stop · +2R break-even · +3R Chandelier 2.5×ATR
                       PositionBook: paper/live positions · stop replacement · 6 %/24 h breaker → flatten, 12 h halt
weekly wfo.py: replay of the logged window over Donchian 16–32 × Chandelier 2.0–3.2 ─► WFE > 0.7 & OOS DD < 10 % gate
               ─► params table (else baseline 20 / 2.5) ─► scanner re-reads every cycle
```

## Run

```bash
make setup                 # python3 -m venv .venv && pip install -r requirements.txt
make test                  # 42 offline tests: indicator math, convex filters, executor vs fake gateway, Kelly/ratchet/breaker, paper + live book, WFO replay ≡ live book, gate, fake /info server end-to-end
make once                  # one cycle, prints the table (first run loads ~2 candle series per coin)
make once ARGS=--convex    # same, plus the breakout engine in dry mode
make run ARGS=--convex     # loop every 60s; Ctrl-C to stop
.venv/bin/python3 squeeze_scanner.py --help
```

Useful flags: `--min-vol/--max-vol` (default 2e6/40e6), `--exclude BTC,ETH,SOL`, `--max-spread-bps 60`,
`--tfs 1h,4h` (the **first** TF gates the engine and the fuel range test), `--equity 5000` (dry-mode equity; live
reads the account), `--risk-pct 1` (bootstrap risk until 20 realized trades) / `--max-risk-pct 4` (Kelly cap),
`--long-only` (default: both sides), `--weight 600` (API budget, see below), `--db state/s38.db` (`--db ''`
disables), `--coins AAA,BBB` (debug: restrict), `--quiet` (no table, log lines only — use under systemd),
`--convex` / `--live` / `--max-positions 3` / `--max-slippage-bps 8` (convex engine), `--scans` / `--events N` /
`--signals N` / `--orders N` / `--positions` / `--trades N` / `--wfo` (print from the db and exit; also the `make`
targets of the same names).

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

The signal carries ATR(14) and the breakout bar's extreme for reference; the stop actually used is the risk
manager's 1.5 × ATR(14) from the fill (next section). Shorts are produced unless `--long-only`.

## Execution protocol (`ConvexExecutor`)

Per signal, in order: kill switch (`STOP` file in the working directory) → account state (live: `clearinghouseState`
equity and open positions; dry: `--equity` and simulated positions) → fresh mids (`allMids`, weight 2) → skip if
already in position, in cooldown (24 bars after a fill, 1 bar after an unfilled/failed attempt), at
`--max-positions`, or if price has already fallen back inside the Donchian range → size → send.

- **Sizing**: from the risk manager — `size = equity × risk_frac / (1.5 × ATR14)`, notional capped at
  `min(equity × 5, 0.05 % of 24h volume)`, floored to `szDecimals`, HL $10 minimum (`too_small or no edge`
  otherwise). Leverage is set per coin to `min(5, maxLeverage)` (cross; isolated for `onlyIsolated` assets) before
  its first order.
- **Entry**: one **IOC limit** at `mid × (1 + 0.08 %)` for buys / `mid × (1 − 0.08 %)` for sells, rounded *toward*
  the mid under HL's tick rule (≤ 5 significant figures, ≤ 6 − szDecimals decimals). Hyperliquid's "market" order
  is exactly this slippage-bounded IOC. Whatever fills within the bound is kept (`filled` / `partial`); the rest is
  cancelled by the exchange and never chased (`unfilled` ⇒ 1-bar cooldown). Realized slippage vs the reference
  mid is recorded in bps.
- **Protection**: immediately after a fill, a **reduce-only stop-market** (`trigger … tpsl: sl, isMarket`) for the
  filled size at `fill ∓ 1.5 × ATR14`, worst acceptable price 5 % beyond the trigger. If the stop is rejected the
  report says `POSITION UNPROTECTED` at CRITICAL level and goes to Telegram. The position is then handed to the
  PositionBook, which owns the exit from there.
- **Dry mode** (default): the exact order payload, sizing and stop are logged and stored with `mode = dry`; the
  fill is simulated at the bound and the paper position is managed exactly like a live one; no exchange request
  is made and the SDK is not even imported.
- **Live mode**: `--live` plus `HL_API_WALLET_KEY` and `HL_ACCOUNT_ADDRESS` (optionally `HL_SUBACCOUNT_ADDRESS`)
  in the environment, otherwise the process refuses to start. Use an API wallet (agent key, trading-only) for a
  dedicated $1k sub-account, never the account's private key. Startup logs the signer and the address being traded
  (both public); the key is never logged or stored. Orders go through the official `hyperliquid-python-sdk`
  (`Exchange.order`, `Exchange.update_leverage`) in a worker thread; exchange requests are weight 1.

## Risk & portfolio management (`risk_manager.py`, `portfolio.py`)

`ConvexRiskManager` is pure logic; `PositionBook` applies it to every open position once per cycle, in paper
(dry) and live mode alike. Target profile this is built for: ~35 % win rate at ≥ 4.5:1 payoff — i.e. most trades
lose small and the book lives off the runners, so the exit logic never takes profit, it only ratchets the stop.

- **Sizing — quarter-Kelly on realized trades.** `f* = p_lb − (1 − p_lb) / b` with `p_lb` the win rate minus one
  standard error (pessimistic) and `b` = mean winning R / mean losing R, both from the `trades` table of the
  current mode; risk fraction = `0.25 · f*`, **capped at 4 % of equity** on the initial stop distance, zero when
  the realized edge is not positive (then nothing is traded). Until **20 trades** exist the bootstrap fraction
  (`--risk-pct`, default 1 %) is used — seeding Kelly with the *target* profile would size at the cap from day one
  on an unvalidated edge. The notional caps from before still apply (≤ 5× equity, ≤ 0.05 % of 24h volume).
  At the target profile the maths gives 3.7 % after 100 trades and hits the 4 % cap after ~300.
- **Initial invalidation stop: 1.5 × ATR(14)** from the actual fill (not the reference price). R = that distance.
  The size is chosen so that the stop loses exactly the risk fraction.
- **Exit ratchet** (stop only ever moves in the trade's favour; evaluated on every new closed bar — the bar's
  adverse extreme is tested against the *previous* stop before its favourable extreme is credited — and on the
  live mark between bars):
  - best excursion ≥ **+2.0R** → stop to **entry + 0.1R** (break-even after taker fees in and out);
  - best excursion ≥ **+3.0R** → **Chandelier**: `best excursion − 2.5 × ATR(14)` (mirror for shorts),
    recomputed with the current ATR each cycle, never lowered.
  Live: when the stop moves ≥ 0.1 % the resting reduce-only stop-market is replaced — new order first, cancel the
  old one second — so the position is never unprotected. A position that disappears from the account is closed from
  its `userFillsByTime` fills (fallback: the stop price) and the leftover stop is cancelled.
- **Circuit breaker — 6 % in 24 h.** Mark-to-market equity (live: `accountValue`; dry: start + realized +
  unrealized − fees) is sampled every cycle; a drawdown ≥ 6 % from the rolling 24 h peak **flattens every position
  (reduce-only IOC, 1 % bound), cancels every open order on the account and halts entries for 12 h**. Cancelling the
  stops without flattening would leave positions naked, hence the flatten. The halt is stored and survives a
  restart; the equity window restarts from the resume point so the same drawdown cannot re-trip immediately.
- **Paper mode** fills at the IOC bound, exits at the stop (or the mark when price has already gapped through it)
  minus 5 bps, charges 4.5 bps taker each side. Trades land in `trades` with R net of fees; `make trades` prints
  them with win rate ± SE, avg R ± SE and payoff, for all trades and per leg. These are the numbers the Kelly
  sizing reads.

Interaction worth knowing: at the 4 % cap, two full-size losses inside 24 h (−8 %) trip the breaker — one and a
half losses is the real daily budget. With the 1 % bootstrap it takes six.

## Trade log & walk-forward optimisation (`wfo.py`, weekly timer)

**Trade log.** Every lifecycle event is written as it happens: `position_events` (open, each stop move with old →
new level and stage, close), `positions` (current state), `trades` (entry, exit, size, duration, entry slippage
vs the reference mid, exit slippage vs the stop, fees, hourly funding paid/received while open from the realized
prints, R net of all of it, exit reason). `make trades` prints the realized statistics.

**Why a replay.** A parameter scan over the Donchian lookback and the Chandelier multiplier cannot be read off the
trade log — a different lookback produces different entries, a different multiplier different exits. `wfo.py`
therefore **replays the live logic** over the logged window, using the log for what it can supply: which coins were
in the universe at each hour (no survivorship bias), the OI samples (HL has no OI history; bars without samples fail
the OI test exactly as live), realized funding, and the realized fee/slippage costs once ≥ 10 trades exist. Entries
use the same filters as `ConvexSignalEngine`, exits call `ConvexRiskManager` itself, so the two cannot drift; every
run also **reconciles** the replay under the live parameters against the logged trades and prints how many it
reproduced.

**Grid and gate.** Donchian 16–32 (step 2) × Chandelier 2.0–3.2 (step 0.2, plus the 2.5 baseline) = 72 points,
each replayed over the last 30 logged days with portfolio limits (3 positions, 24-bar cooldown) but without the
breaker (a kill switch is not a parameter). Trades belong to in-sample (first 20 days) or out-of-sample (last 10)
by entry time. Sharpe is annualised from daily R sums (days without trades = 0) with its standard error; max
drawdown compounds the book's current risk fraction per trade. A non-baseline point is adopted only if **all** of:
`n_IS ≥ 30`, `n_OOS ≥ 10`, `Sharpe_IS > 0`, **WFE = Sharpe_OOS / Sharpe_IS > 0.70**, **OOS max drawdown < 10 %**,
and its OOS Sharpe is not below the baseline's. Otherwise the **baseline (20, 2.5)** is (re)installed — a degraded
regime therefore falls back by construction. Picking the best of 72 points inflates the in-sample number, which is
why the WFE gate exists and why the report shows the standard errors.

**Mechanics.** `deploy/s38-wfo.timer` runs `wfo.py` Mondays 00:40 UTC; the verdict and the full grid go to
`wfo_runs`, the chosen values to `params`, and the running scanner re-reads `params` every cycle (open positions
keep ratcheting with the new multiplier; the stop never moves against them). `make wfo` prints the same report
without touching anything; `make wfo-apply` is what the timer does. Until ~30 days and ≥ 30 in-sample trades are
logged the job says so and stays at baseline — at the signal rate observed on day one that is months away, which is
itself a finding.

## Output

Table legend is printed under each table. sqlite tables (`state/s38.db`): `scans` (one row per coin per cycle:
price, funding, OI in coin units, volume, spread, both TFs' squeeze metrics, fuel, state, score), `events` (every
state transition, FIRE rows carry the intent JSON), `funding_hourly` (realized hourly prints), `universe`
(eligibility counts), `signals` (one row per evaluated breakout bar with every filter value, `accepted`, `reasons`),
`orders` (every execution report: plan, fill, slippage, stop, raw exchange response), `positions` (open and closed,
with stop stage, best excursion, stop oid), `position_events` (open / stop_move / close as they happen), `trades`
(closed trades: entry, exit, duration, slippage, fees, funding, R net, reason), `breaker` (trips), `params` (active
WFO-tuned values), `wfo_runs` (every weekly run with its full grid), `candles` (1h cache for the replay). Times are
UTC ms. Views: `make scans | events | signals | orders | positions | trades | wfo`.

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
itself on first start; the installer also enables the weekly `s38-wfo.timer` — `systemctl list-timers s38-wfo.timer`).

## Known limits

- HIP-3 builder dexes are not scanned (main perp universe only).
- `funding` from `metaAndAssetCtxs` is the current hour's rate and moves within the hour; the realized prints
  are in `funding_hourly`.
- Signal latency = the cycle that follows the hourly close: 1–3 min at 600 weight/min, up to ~7 min at 400 when
  the 1h and 4h refresh coincide. Entries are therefore minutes after the close, never intra-bar.
- HL has no OI history endpoint; OI confirmation only works from the scanner's own samples (≥ 1 h of uptime).
- Exits are stop-based only (invalidation → break-even → Chandelier). No take-profit by design, no time stop.
- Paper fills assume the whole size fills at the IOC bound; thin books will do worse live.
- Nothing here is a backtest. The squeeze → expansion thesis, the volume/OI filters, the stop placement and the
  35 % / 4.5:1 target profile are all unvalidated on HL mid-caps until the study below is done. Kelly sizing only
  engages once 20 realized trades exist and switches itself off if the realized edge is not positive.

## Graduation

1. Run `--convex` in dry mode on hospitality1 and let `signals` / `orders` / `trades` accumulate. Review with
   `make signals`, `make orders`, `make positions`, `make trades`: how many breakouts pass each filter, where the
   paper fills were, how the ratchet exited them, and the realized win rate / payoff with standard errors.
2. **Historical study** (proposed, not started): squeeze releases and Donchian/volume/OI breakouts on the current
   universe from `candleSnapshot` (up to 5000 1h bars ≈ 200 days per coin): forward |return| and signed return at
   4/12/24/48 bars, split by direction, by fuel / no fuel and by 4h confirmation; n, mean, median, SE, skew, hit rate
   per leg against unconditional bars of the same coins. OI cannot be backtested (no history endpoint), so the OI
   filter's marginal value can only be measured live from the dry log.
3. Only if 1–2 hold up: `--live` on a dedicated $1k sub-account driven by an API wallet, `--max-positions 3`,
   1 % bootstrap risk; Kelly takes over from the 21st live trade.
