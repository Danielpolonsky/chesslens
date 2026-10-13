"""Build the precomputed demo database and the ML model from public Lichess data.

Data source: Lichess open database, January 2013 standard rated games
(https://database.lichess.org/, released under CC0 / public domain).

    python scripts/prepare_demo.py                     # full build (~10 min on 8 cores)
    python scripts/prepare_demo.py --skip-corpus       # demo database only

Outputs
  data/demo/demo.db            demo player's games, fully analysed (committed)
  data/cache/ml_corpus.db      other players' games used to train the model (not committed)
  data/models/ml_metrics.json  held-out evaluation results (committed)
  data/models/risk_model.joblib
"""

from __future__ import annotations

import argparse
import io
import random
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src import database as db  # noqa: E402
from src.engine_analysis import StockfishAnalyzer  # noqa: E402
from src.features import build_ml_frame  # noqa: E402
from src.ingest import parse_pgn_text  # noqa: E402
from src import ml  # noqa: E402

LICHESS_URL = "https://database.lichess.org/standard/lichess_db_standard_rated_2013-01.pgn.zst"
RAW_PATH = config.DATA_DIR / "raw" / "lichess_2013-01.pgn.zst"
CORPUS_DB = config.DATA_DIR / "cache" / "ml_corpus.db"
# Chosen with the documented criteria in README ("Dataset"): 60-160 games in the month,
# average rating 1350-1850 and a mix of time controls.
DEFAULT_PLAYER = "sani5057"
SOURCE_LABEL = "lichess_db_2013_01"


def download(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {LICHESS_URL} ...")
    with requests.get(LICHESS_URL, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(path, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)


def read_game_chunks(path: Path) -> list[str]:
    import zstandard

    with open(path, "rb") as fh:
        text = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(fh), encoding="utf-8").read()
    return [c if c.startswith("[Event") else "[Event" + c for c in text.split("\n\n[Event") if c.strip()]


def header(chunk: str, tag: str) -> str | None:
    m = re.search(rf'^\[{tag} "([^"]*)"\]', chunk, flags=re.M)
    return m.group(1) if m else None


def _analyze_chunk(args):
    """Worker: analyse a batch of games with its own Stockfish process."""
    games, depth = args
    out = []
    with StockfishAnalyzer(depth=depth) as analyzer:
        for game_id, fens, ucis in games:
            out.append((game_id, analyzer.analyze_moves(fens, ucis), analyzer.depth, analyzer.engine_name))
    return out


def analyze_db_parallel(db_path: Path, depth: int, workers: int, batch: int = 8) -> None:
    conn = db.connect(db_path)
    ids = [r[0] for r in conn.execute(
        "SELECT game_id FROM games WHERE game_id NOT IN (SELECT game_id FROM analysis_runs)")]
    jobs, move_ids = [], {}
    for gid in ids:
        rows = db.game_moves(conn, gid)
        move_ids[gid] = [r["move_id"] for r in rows]
        jobs.append((gid, [r["fen_before"] for r in rows], [r["uci"] for r in rows]))
    batches = [(jobs[i:i + batch], depth) for i in range(0, len(jobs), batch)]
    print(f"  analysing {len(jobs)} games at depth {depth} with {workers} workers ...")
    start, done = time.time(), 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_analyze_chunk, b) for b in batches]
        for fut in as_completed(futures):
            for gid, evals, d, engine in fut.result():
                db.save_game_analysis(conn, gid, move_ids[gid], evals, d, engine)
                done += 1
            print(f"    {done}/{len(jobs)} games ({time.time() - start:.0f}s)", flush=True)
    conn.close()


def build_db(db_path: Path, chunks: list[str]) -> None:
    if db_path.exists():
        db_path.unlink()
    report = parse_pgn_text("\n\n".join(chunks), source=SOURCE_LABEL)
    print(f"  parsed {len(report.games)} games, skipped {report.skipped}")
    conn = db.connect(db_path)
    summary = db.insert_games(conn, report.games)
    print(f"  inserted {summary.inserted}, duplicates {summary.duplicates}")
    conn.close()


def finalize(db_path: Path) -> None:
    """Checkpoint WAL into the main file so the database is a single portable file."""
    conn = db.connect(db_path)
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.execute("PRAGMA journal_mode=DELETE")
    conn.execute("VACUUM")
    conn.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--player", default=DEFAULT_PLAYER)
    ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--corpus-games", type=int, default=800)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-corpus", action="store_true")
    ap.add_argument("--skip-demo", action="store_true")
    args = ap.parse_args()

    download(RAW_PATH)
    print("Reading Lichess file ...")
    chunks = read_game_chunks(RAW_PATH)
    print(f"  {len(chunks)} games in file")

    player = args.player.lower()
    focal = [c for c in chunks if player in {(header(c, "White") or "").lower(),
                                             (header(c, "Black") or "").lower()}]
    if not args.skip_demo:
        print(f"Building demo database for {args.player} ({len(focal)} games)")
        build_db(config.DEMO_DB_PATH, focal)
        analyze_db_parallel(config.DEMO_DB_PATH, args.depth, args.workers)
        finalize(config.DEMO_DB_PATH)

    if args.skip_corpus:
        return

    def eligible(c: str) -> bool:
        if player in {(header(c, "White") or "").lower(), (header(c, "Black") or "").lower()}:
            return False
        if not (header(c, "WhiteElo") or "").isdigit() or not (header(c, "BlackElo") or "").isdigit():
            return False
        tc = header(c, "TimeControl") or "-"
        if tc == "-":
            return False
        return len(re.findall(r"\d+\.", c.split("\n\n", 1)[-1])) >= 15  # >= ~15 full moves

    pool = [c for c in chunks if eligible(c)]
    rng = random.Random(args.seed)
    sample = rng.sample(pool, min(args.corpus_games, len(pool)))
    print(f"Building ML corpus ({len(sample)} of {len(pool)} eligible games)")
    CORPUS_DB.parent.mkdir(parents=True, exist_ok=True)
    build_db(CORPUS_DB, sample)
    analyze_db_parallel(CORPUS_DB, args.depth, args.workers)

    print("Training models ...")
    corpus_conn = db.connect(CORPUS_DB)
    frame = build_ml_frame(ml.load_ml_moves(corpus_conn))
    corpus_conn.close()
    demo_conn = db.connect(config.DEMO_DB_PATH, read_only=True)
    external = build_ml_frame(ml.load_ml_moves(demo_conn))
    demo_conn.close()
    result = ml.train_and_evaluate(frame, seed=args.seed, external=external)
    result.metrics["data_source"] = LICHESS_URL
    result.metrics["engine_depth"] = args.depth
    ml.save_artifacts(result, config.ML_METRICS_PATH, config.ML_MODEL_PATH)
    frame.to_csv(config.MODELS_DIR / "ml_training_frame.csv.gz", index=False, compression="gzip")
    m = result.metrics
    print(f"  selected: {m['selected_model']}  positions={m['n_positions']}  games={m['n_games']}")
    for name, r in m["models"].items():
        t = r["test"]
        print(f"  {name:22s} AP={t['average_precision']:.3f} AUC={t['roc_auc']:.3f} "
              f"P={t['precision']:.3f} R={t['recall']:.3f} F1={t['f1']:.3f}")


if __name__ == "__main__":
    main()
