# Detecting Anomalies and Structural Breaks in Chinese Macroeconomic Indicators

A small research project that takes eight Chinese macroeconomic series, cleans them,
and then applies three independent anomaly-detection methods plus a structural-break
test to ask a simple question: **which movements in these indicators are genuinely
unusual, and which are just noise?**

## What this project does

1. Pulls monthly macro indicators from the Wind terminal (M0/M1/M2 growth, CPI, PPI,
   manufacturing PMI, total social financing, industrial production).
2. Cleans them into a consistent month-end, gap-filled-as-NaN panel and derives the
   M1/M2 growth gap.
3. Runs four detection layers of increasing complexity:
   - rolling-window Z-score (statistical, univariate)
   - STL decomposition with residual testing (removes seasonality before flagging)
   - Isolation Forest (machine learning, multivariate)
   - structural-break detection (tests whether the series' mean level shifted)
4. Evaluates the results against known economic shocks, checks agreement between the
   three methods, and runs a parameter-sensitivity test.

## Why it matters

Macro data is published monthly and every release produces commentary about whether
the number was "unexpected". Most of those movements are statistically unremarkable.
Telling a real signal apart from a plausible artefact is the same problem that sits
behind risk identification in finance and, for that matter, behind crime forecasting:
in each case the task is to decide, from a large and imperfect dataset, which
observations actually matter.

## Data

Six monthly series, obtained from the East Money public data centre
(`datacenter-web.eastmoney.com`), which republishes the PBoC and NBS releases.

| Indicator | East Money field | From |
|---|---|---|
| M2 growth (YoY) | `BASIC_CURRENCY_SAME` | 2008-01 |
| M1 growth (YoY) | `CURRENCY_SAME` | 2008-01 |
| M0 growth (YoY) | `FREE_CASH_SAME` | 2008-01 |
| CPI (YoY) | `NATIONAL_SAME` | 2008-01 |
| PPI (YoY) | `BASE_SAME` | 2006-01 |
| Manufacturing PMI | `MAKE_INDEX` | 2008-01 |

Sample period: **January 2008 – August 2026** (224 months), with PPI available from
January 2006. The extracted raw data is committed to `data/raw/macro_raw.csv` so the
analysis can be reproduced without any data subscription.

**A note on the field names.** They are misleading and must not be taken at face value.
Verifying against the PBoC's August 2026 release (M2 balance ¥356.81tn, +7.5%; M1
¥115.77tn, +4.1%; M0 ¥14.83tn, +11.2%) shows that `BASIC_CURRENCY` holds **M2**,
`CURRENCY` holds **M1**, and `FREE_CASH` holds **M0**. Mapping them by name produces a
series with large spurious jumps. The mapping used here is documented in
`src/fetch_public.py` alongside the figures it was checked against.

A parallel script, `src/fetch_wind.py`, obtains the same indicators from the Wind
terminal where a licence is available. Everything downstream of `data/raw/macro_raw.csv`
is source-agnostic.

## Method

### Layer 1 — Rolling-window Z-score

For each month, the Z-score is computed against the mean and standard deviation of
the preceding 36 months:

```
z_t = (x_t − mean(x_{t−36 : t−1})) / std(x_{t−36 : t−1})
```

An observation is flagged when `|z| > 2.5`.

The window is rolled rather than computed over the full sample on purpose. Using the
full sample mean would let the 2008 value be judged using 2009–2025 data, which is
not available at the time. A rolling window uses only information that existed then.

### Layer 2 — STL decomposition and residual testing

Monthly macro series carry strong seasonality (Chinese New Year, quarter-ends). A
plain Z-score on the raw series flags the same months every year. This layer
decomposes the series first:

```
x_t = Trend_t + Seasonal_t + Residual_t
```

and tests the residual for outliers, using a robust scale estimate (1.4826 × MAD)
so that an extreme value does not inflate the threshold that is meant to catch it.

### Layer 3 — Isolation Forest

The first two layers look at one series at a time. This layer looks at all of them
together, which is the only way to catch a period that is unremarkable in each
individual indicator but unusual in combination. Features are the current values plus
month-on-month differences, standardised before fitting.

### Layer 4 — Structural breaks

Anomaly detection finds points that deviate. A structural break is different: it says
the series' behaviour has changed, so past regularities no longer hold. This layer
compares the mean of the 36 months before each point with the mean of the 36 months
after, using Welch's t-test, and keeps breaks significant at *p* < 0.001. Consecutive
months in the same break are collapsed to the single largest shift.

**This is a simplified approach, not a formal break test.** See *Limitations*.

## Results

Run `python src/run_all.py` to regenerate everything below. Numbers refer to the
committed dataset: **January 2008 – August 2026, 224 monthly observations** for M0/M1/M2,
CPI and PMI, and **January 2006 – August 2026** for PPI.

### What I found

**1. All six known shocks were detected, and the two largest were caught by all three
methods independently.** Checking each event against a ±3-month window:

| Event | Nearby anomalies | Detected by |
|---|---|---|
| 2008-09 Global financial crisis | 6 | STL, Z-score |
| 2008-11 Four-trillion stimulus | 7 | **all three** |
| 2015-06 A-share crash | 6 | STL, Z-score |
| 2020-02 COVID-19 shock | 6 | STL, Z-score |
| 2020-04 Post-COVID monetary easing | 6 | STL, Z-score |
| 2022-04 Shanghai lockdown | 1 | **all three** |

The two events where the multivariate layer agreed are the two that hit several
indicators at once rather than moving one series at a time. That is what the
multivariate layer is for, and it is the clearest evidence in this project that the
three layers are not redundant.

**2. The M1/M2 growth-rate gap has had three regime changes, not one.** The structural
break test finds the gap shifting between persistent regimes rather than returning to a
single norm:

| Period | Mean gap | Break detected |
|---|---|---|
| 2008–2011 | −0.81 | — |
| 2012–2016 | −2.82 | 2011-12: −0.28 → −6.96 |
| 2017–2019 | −0.50 | 2014-12: −6.96 → +4.05 |
| 2020–2026 | −5.67 | 2019-01 / 2020-01: +4.78 → −3.96 → −4.39 |

The 2011 break coincides with the credit tightening that followed the stimulus period.
The 2014–16 break coincides with the 2015 rate cuts and the local government debt swap.
The most recent regime, starting around 2019–20, is the only one in the sample where the
gap has stayed negative for six consecutive years. **This is the finding I would take
furthest: for the past six years, M2 has consistently grown faster than M1, which is the
opposite of the pattern in 2008–2011.**

**3. The three-way agreement is concentrated in a specific and interpretable window.**
Only 6 of 139 flagged months were flagged by all three methods, and Feb–Mar 2009 is the
single most unusual period in the sample — CPI was negative (−1.56), PPI was falling,
and the M1/M2 gap was at −9.61, all simultaneously.

**4. De-seasonalisation removed most of the seasonal false positives — but not the
Chinese New Year effect.** Of the time points flagged by all three methods, 18 of 31 fall
in January or February. This is larger than a uniform distribution would predict
(2/12 = 17% would give roughly 5). The mechanism is visible in the raw data: M0 growth
runs at +31.21% in January 2008 and +5.96% in February 2008, because the timing of
Chinese New Year determines which month the cash injection lands in. STL removes a fixed
12-month seasonal pattern; it cannot remove a holiday whose date moves within a
two-month window each year. **See Limitation 7 — this is a real weakness of the current
design, not a cosmetic issue.**

### Agreement between methods

| Methods agreeing | Time points |
|---|---|
| 3 of 3 | 6 |
| 2 of 3 | 48 |
| 1 of 3 | 85 |

### Known-event check

6 of 6 events detected within ±3 months (100%). Note that this measures recall only —
see Limitation 1.

### Sensitivity

For M1 growth, changing the window from 24 to 48 months and the threshold from 2.0 to
3.0 gives:

| Threshold | Anomalies (24m) | (36m) | (48m) | Verdict |
|---|---|---|---|---|
| \|z\| > 2.0 | 31 | 30 | 27 | stable |
| \|z\| > 2.5 | 12 | 12 | 13 | stable |
| \|z\| > 3.0 | 4 | 6 | 8 | spread of 2× |

At thresholds of 2.0 and 2.5 the count barely moves with window length, which is
reassuring. At 3.0 the count doubles from 4 to 8 — with such small counts, a change of
four months is a large relative shift. **I would not rely on results at |z| > 3.0.**

### Figures

| File | Content |
|---|---|
| `figures/fig1_main.png` | Main series with anomalies from Z-score and STL |
| `figures/fig1b_iforest.png` | Multivariate anomalies, shown as bands |
| `figures/fig2_scissors.png` | M1/M2 growth-rate gap with structural breaks |
| `figures/fig3_methods.png` | Per-year detection counts and method agreement |
| `figures/fig4_sensitivity.png` | Anomaly count vs. window length |

## Limitations

Being explicit about these is more useful than hiding them:

1. **No ground truth.** There is no labelled dataset of "true" macro anomalies. The
   known-event check measures whether the methods catch shocks we already know about;
   it says nothing about how many false positives they produce. Precision cannot be
   estimated from this design.
2. **The structural-break test is a simplification.** Comparing means before and after
   each candidate point is a screening device, not a formal test. A proper treatment
   would use Bai–Perron or a Chow test with known break dates. The current approach
   will over-detect when the series has a smooth trend.
3. **Sample period restrictions.** The series begin in 2008 (PPI in 2006) because that is
   where the public source's coverage starts. This excludes both the 1997 Asian financial
   crisis and the 2003 SARS period, and it means the model has never seen a full
   pre-2008 credit cycle.
4. **Multicollinearity in the Isolation Forest.** M0/M1/M2 growth rates are strongly
   correlated. The model may be weighting essentially the same signal several times.
   A principal-component step would be the natural fix.
5. **No out-of-sample test.** All detection is in-sample. A stronger design would fit
   on an early period and evaluate on a later one.
6. **Data revisions.** Macro series are revised. The CSV committed here is a snapshot of
   the public source as of August 2026; re-running `fetch_public.py` later may produce
   slightly different historical values.
7. **The Chinese New Year effect is not removed.** This is the most serious weakness in
   the current design. STL assumes a fixed 12-month period, but Chinese New Year moves
   between late January and mid-February, so the cash-demand distortion moves with it.
   The consequence is measurable: of the 31 time points flagged by the Isolation Forest,
   18 fall in January or February, against roughly 5 expected under a uniform
   distribution. A proper fix would regress each January/February observation on the
   number of pre-holiday working days in that month, which is a clearly defined next
   step rather than something the current pipeline can address.

## Reproducing

```bash
# 1. environment
pip install -r requirements.txt

# 2. data
#    Option A - use the committed snapshot (no network, no subscription):
#        data/raw/macro_raw.csv is already in the repo; skip to step 3.
#    Option B - re-fetch from the public source:
python src/fetch_public.py
#    Option C - fetch from the Wind terminal (requires a licence and the
#               terminal running and logged in):
python src/fetch_wind.py

# 3. pipeline
python src/clean.py        # clean + derive indicators
python src/detect.py       # four detection layers
python src/evaluate.py     # known-event check, agreement, sensitivity
python src/plot.py         # figures

# or all at once:
python src/run_all.py --no-fetch
```

## Repository layout

```
macro-anomaly-detection/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/                 Wind export (committed for reproducibility)
│   └── processed/           cleaned panels, features, known events
├── src/
│   ├── fetch_public.py      data acquisition (East Money public API)
│   ├── fetch_wind.py        data acquisition via WindPy (optional)
│   ├── clean.py             cleaning and derived indicators
│   ├── detect.py            four detection layers
│   ├── evaluate.py          evaluation and sensitivity
│   ├── plot.py              figures
│   └── run_all.py           run the whole pipeline
├── results/                 anomaly tables
└── figures/                 output charts
```

## Environment

Tested with Python 3.11+, pandas 2.x, scikit-learn 1.3+, statsmodels 0.14+.
