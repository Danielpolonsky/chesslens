# ChessLens: Resume and Interview Materials

Every number below was measured in this repository (see `README.md` and `data/models/ml_metrics.json`).
The app is **not yet deployed**; add the live URL here once it is.

## Project title

**ChessLens: Stockfish-Powered Chess Analytics Platform** (Python, SQL, machine learning), built for Polonsky Academy

## Tech stack

Python 3.12 · Streamlit · SQLite (normalized schema, views, CTE analytics) · Pandas · NumPy ·
python-chess · Stockfish · Plotly · scikit-learn · pytest · Playwright · Git

## Resume bullets

- Built a full-stack chess analytics app in Python/Streamlit with this pipeline: PGN and
  Chess.com API import, a Stockfish evaluation of every position, normalized SQLite storage with
  content-hash deduplication, then SQL/Pandas analytics. Ships with 158 pre-analysed public games
  (9,513 plies) and analyses a fresh 12-game upload in about 8 seconds at depth 10.
- Designed SQL analytics (a perspective view plus CTE queries) for centipawn loss, error rates by
  phase, colour, opening and time control, with Wilson intervals. Built a rule-based coaching
  engine that ranks weaknesses by two-proportion z statistics and links each one to real example
  positions.
- Trained a serious-mistake risk model on 56,484 positions from 800 games. Used pre-move features
  only, with an explicit leakage check, a game-grouped train/test split and grouped CV. The random
  forest reached an average precision of 0.30 against a baseline of 0.19 (ROC AUC 0.69, top-decile
  lift 1.9×), and the limitations are documented honestly. The project is backed by 87 automated
  tests.

**Shorter variant (one line):**
Built ChessLens, a Streamlit + SQLite + Stockfish chess analytics app with SQL-driven performance
insights, statistically ranked coaching recommendations and a game-grouped ML risk model
(AP 0.30 vs 0.19 baseline); 87 pytest tests.

## Real project statistics

| Metric | Value |
| --- | --- |
| Application code (non-blank lines, `app.py` + `src/`) | ~3,500 |
| Automated tests (pytest) | 87, all passing locally; plus a Playwright browser smoke test |
| Demo dataset | 158 public Lichess games (Jan 2013, CC0), 9,513 plies analysed at depth 10 |
| Demo player summary | 65 W / 6 D / 87 L, average CPL 60, blunder rate 8.1% |
| ML corpus | 800 games, 56,484 positions, 18.7% serious-error base rate (test set) |
| ML split | 600 train / 200 test games (grouped by game), 5-fold GroupKFold CV |
| Random forest, test set | AP 0.301, ROC AUC 0.686, precision 0.271, recall 0.708, F1 0.392 |
| Logistic regression, test set | AP 0.279, ROC AUC 0.656 |
| Dummy baseline, test set | AP 0.185, ROC AUC 0.495 |
| External check (demo player, unseen) | AP 0.285, ROC AUC 0.676, top-decile lift 1.88× |
| In-app analysis speed | 12-game PGN analysed in ~8 s at depth 10 (laptop, 1 engine thread) |

## GitHub repository description

> Stockfish-powered chess analytics in Streamlit: import PGN or Chess.com games, find where you
> lose centipawns by phase, opening and time control, get statistically ranked coaching priorities,
> and explore a leakage-safe mistake-risk model. Python, SQLite, Plotly, scikit-learn.

Suggested topics: `chess`, `stockfish`, `streamlit`, `sqlite`, `data-analytics`, `scikit-learn`, `python`

## 60-second pitch

> "ChessLens is a chess analytics app I built for Polonsky Academy. You upload a PGN or type a
> Chess.com username, and it runs every position through Stockfish to measure how many centipawns
> each move loses. It labels each move as an inaccuracy, mistake or blunder. Everything goes into a
> normalized SQLite database, and the dashboard is driven by SQL queries. It shows where you
> actually lose games: by phase, colour, opening and time control, with confidence intervals so
> small samples aren't over-interpreted. The coaching page ranks your top three weaknesses by a
> z statistic and links each one to real positions from your games, shown on a board with the
> engine's better move. I also trained a model that predicts mistake risk from the position before
> a move. I split by game to avoid leakage, and it roughly doubles the hit rate in the riskiest
> positions. It's modest, and I document that honestly. The app has 87 automated tests and keeps
> each user's uploads private to their session."

## Interview questions and answers

**1. How do you compute centipawn loss, and what are the edge cases?**
I evaluate the position before and after every move with Stockfish and convert both scores to the
mover's point of view; CPL is `max(0, before − after)`. The edge cases:
- *Mates:* I map mate scores to ±10,000 and then clamp all evaluations to ±1,000. Without the clamp,
  missing a mate-in-5 when you're already +15 would count as thousands of centipawns.
- *Best moves:* if the played move is the engine's best, CPL is 0, to avoid search noise.
- *Game-ending positions:* checkmate and stalemate are scored exactly, without the engine.

I evaluate N+1 positions per game rather than 2N by reusing each "after" evaluation as the next
move's "before". A per-position cache means repeated opening positions are evaluated only once.

**2. How did you prevent data leakage in the ML model?**
There are two kinds of leakage.
- *Feature leakage:* the target is derived from the engine's evaluation of the played move, so any
  engine output or move identity would leak the label. I only use features computed from the board
  *before* the move, and a check raises an error if columns like `cpl`, `eval_after` or
  `best_move_uci` ever reach the feature matrix.
- *Split leakage:* positions from the same game are highly correlated. I split with
  `GroupShuffleSplit` by game and did model selection with `GroupKFold`, so no game contributes to
  both training and evaluation. A test asserts that the split has no overlap.

I also evaluated on a completely separate player's games as an external check.

**3. Your model's average precision is 0.30. Is that good?**
It's meaningfully better than the 0.19 baseline: the riskiest 10% of positions contain serious
errors at about 1.9 times the base rate, and that held up on an unseen player. But it's weak in
absolute terms. Precision is around 27%, and because of balanced class weights the raw
probabilities are inflated, so I present it as a risk ranking, not a predictor. Mistakes depend on
concrete tactics and calculation that my static features don't capture. I'd rather report that
honestly than tune a threshold until a single metric looks impressive.

**4. Why SQLite, and how is the database designed?**
SQLite gives real SQL with zero infrastructure, which suits a single-server Streamlit app. Each
browser session gets its own database file for privacy, and the demo database is opened read-only.
The schema is normalized:
- players and games, with a unique content hash for deduplication;
- one row per ply in `moves`, holding the position and pre-move features;
- engine results in `move_analysis`;
- a `position_evals` table that doubles as the engine cache.

A `v_player_games` view presents each game from each player's perspective. That turns questions
like "score as Black in blitz" into simple filters, and the analytics are composed from CTEs over it.

**5. How do the coaching recommendations work, and how do you avoid overclaiming?**
They're rule-based, not generated text. Each candidate weakness compares one group of the player's
moves with the rest using a two-proportion z statistic (or a score-difference z for results). The
candidates are: the weakest phase, colour, an opening with 5+ games, converting winning positions,
time pressure, late-game errors and time control. I rank by z, avoid repeating the same type
(for example, two openings), and require minimum sample sizes. Each card shows the actual rates, the sample size, a confidence
label and real example positions. The app explicitly doesn't claim tactical motif detection such
as forks or pins, because I didn't implement it. The advice only refers to things the system
actually measures.
