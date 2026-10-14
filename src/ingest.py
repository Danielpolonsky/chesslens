"""Import -> Parse -> Clean -> Validate.

Turns PGN text (from an upload or the Chess.com public API) into validated
``ParsedGame`` objects with per-move, pre-move board features.
"""

from __future__ import annotations

import hashlib
import io
import logging
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Iterable

import chess
import chess.pgn
import requests

from . import config
from .features import position_features

VALID_RESULTS = {"1-0", "0-1", "1/2-1/2"}
OPENING_KEYWORDS = ("Defense", "Defence", "Opening", "Game", "Gambit", "Attack", "System")
USERNAME_RE = re.compile(r"^[A-Za-z0-9_-]{2,30}$")
SUPPORTED_VARIANTS = {"standard", "chess", "from position"}

# python-chess logs every rejected token; rejections are reported to the user instead.
logging.getLogger("chess.pgn").setLevel(logging.CRITICAL)

# Rejection reasons shown in the import summary.
REASON_ILLEGAL = "illegal move"
REASON_UNREADABLE = "unreadable move text"
REASON_VARIANT = "unsupported variant"
REASON_EMPTY = "no moves"
REASON_FEN = "invalid starting position (FEN)"
REASON_ERROR = "could not be processed"


@dataclass
class ParsedMove:
    ply: int
    move_number: int
    color: str
    san: str
    uci: str
    fen_before: str
    clock_before_s: float | None
    features: dict


@dataclass
class ParsedGame:
    white: str
    black: str
    white_elo: int | None
    black_elo: int | None
    result: str
    played_at: str | None
    time_control: str | None
    time_class: str
    eco: str | None
    opening: str | None
    opening_family: str
    termination: str | None
    site: str | None
    source: str
    moves: list[ParsedMove] = field(default_factory=list)
    pgn: str = ""

    @property
    def game_hash(self) -> str:
        key = "|".join(
            [
                self.white.lower(),
                self.black.lower(),
                self.played_at or "",
                self.result,
                " ".join(m.uci for m in self.moves),
            ]
        )
        return hashlib.sha1(key.encode("utf-8")).hexdigest()

    @property
    def num_plies(self) -> int:
        return len(self.moves)


@dataclass
class ParseReport:
    games: list[ParsedGame] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)  # file-level problems and per-game rejections
    skipped: int = 0
    truncated: bool = False
    games_read: int = 0
    skip_reasons: Counter = field(default_factory=Counter)
    duplicates_in_file: int = 0
    notes: list[str] = field(default_factory=list)

    def reject(self, index: int, reason: str, detail: str = "") -> None:
        self.skipped += 1
        self.skip_reasons[reason] += 1
        self.errors.append(f"Game {index}: {reason}" + (f" ({detail})" if detail else "") + ".")

    def merge(self, other: "ParseReport") -> None:
        self.errors.extend(other.errors)
        self.skipped += other.skipped
        self.games_read += other.games_read
        self.skip_reasons.update(other.skip_reasons)
        self.duplicates_in_file += other.duplicates_in_file


class GameRejected(ValueError):
    def __init__(self, reason: str, detail: str = ""):
        super().__init__(detail or reason)
        self.reason = reason
        self.detail = detail


# ---------------------------------------------------------------------------
# Header cleaning helpers
# ---------------------------------------------------------------------------


def clean_name(value: str | None) -> str:
    value = (value or "").strip()
    return value if value and value != "?" else "Unknown"


def parse_elo(value: str | None) -> int | None:
    if value is None:
        return None
    value = value.strip()
    if not value.isdigit():
        return None
    elo = int(value)
    return elo if 100 <= elo <= 3500 else None


def parse_pgn_date(date: str | None, time_str: str | None = None) -> str | None:
    """Convert PGN ``YYYY.MM.DD`` (+ optional ``HH:MM:SS``) to ISO. Unknown -> None."""
    if not date:
        return None
    parts = date.strip().replace("-", ".").split(".")
    if len(parts) != 3 or not all(p.isdigit() for p in parts):
        return None
    try:
        day = datetime(int(parts[0]), int(parts[1]), int(parts[2]))
    except ValueError:
        return None
    if time_str and re.fullmatch(r"\d{1,2}:\d{2}:\d{2}", time_str.strip()):
        h, m, s = (int(x) for x in time_str.strip().split(":"))
        if h < 24 and m < 60 and s < 60:
            return day.replace(hour=h, minute=m, second=s).strftime("%Y-%m-%d %H:%M:%S")
    return day.strftime("%Y-%m-%d")


def classify_time_control(tc: str | None) -> str:
    """Lichess-style buckets on estimated duration = base + 40 * increment."""
    if not tc or tc in {"?", ""}:
        return "unknown"
    if tc == "-" or "/" in tc:
        return "daily"
    try:
        base, _, inc = tc.partition("+")
        total = float(base) + 40 * float(inc or 0)
    except ValueError:
        return "unknown"
    if total < 179:
        return "bullet"
    if total < 479:
        return "blitz"
    if total < 1499:
        return "rapid"
    return "classical"


def opening_from_eco_url(url: str | None) -> str | None:
    """Chess.com ``ECOUrl`` -> readable opening name (move suffixes removed)."""
    if not url or "/openings/" not in url:
        return None
    slug = url.rstrip("/").split("/openings/")[-1]
    words = []
    for token in slug.split("-"):
        if not token or token[0].isdigit():
            break
        words.append(token)
    name = " ".join(words).strip()
    return name or None


def opening_family(name: str | None, eco: str | None = None) -> str:
    """Collapse detailed opening names to a family for meaningful sample sizes."""
    if name:
        head = name.split(":")[0].split(",")[0].strip()
        words = head.split()
        for i, word in enumerate(words):
            if word in OPENING_KEYWORDS:
                return " ".join(words[: i + 1])
        return " ".join(words[:3]) if words else "Unknown"
    if eco and re.fullmatch(r"[A-E]\d\d", eco):
        return f"ECO {eco[0]}"
    return "Unknown"


# ---------------------------------------------------------------------------
# PGN parsing
# ---------------------------------------------------------------------------


_HEADER_LINE_RE = re.compile(r'^\s*\[\w+\s+"')
_COMMENT_RE = re.compile(r"\{[^}]*\}")
_LINE_COMMENT_RE = re.compile(r";[^\n]*")
_NAG_RE = re.compile(r"\$\d+")
_MOVE_NUMBER_RE = re.compile(r"^\d+\.+")
_ANNOTATION_RE = re.compile(r"[!?]+$")
_RESULT_TOKENS = {"1-0", "0-1", "1/2-1/2", "*"}
_CASTLING_TOKENS = {"O-O", "O-O-O", "0-0", "0-0-0", "--"}


def mainline_tokens(raw_game: str) -> list[str]:
    """Move tokens of a game's main line (comments, variations, NAGs, numbers removed).

    python-chess silently ignores tokens it cannot read (e.g. ``Zz9``), which would make
    a damaged game look complete. Comparing this count with the parsed move count
    exposes such games.
    """
    body = "\n".join(
        line for line in raw_game.splitlines()
        if not _HEADER_LINE_RE.match(line) and not line.startswith("%")
    )
    body = _LINE_COMMENT_RE.sub(" ", _COMMENT_RE.sub(" ", body))
    kept, depth = [], 0
    for ch in body:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        elif depth == 0:
            kept.append(ch)
            continue
        kept.append(" ")
    tokens = []
    for tok in _NAG_RE.sub(" ", "".join(kept)).split():
        tok = _ANNOTATION_RE.sub("", _MOVE_NUMBER_RE.sub("", tok))
        if tok and tok not in _RESULT_TOKENS:
            tokens.append(tok)
    return tokens


def _first_unreadable(tokens: list[str]) -> str | None:
    return next((t for t in tokens if t not in _CASTLING_TOKENS and not chess.SAN_REGEX.match(t)), None)


def _game_to_parsed(game: chess.pgn.Game, source: str, raw: str | None = None) -> ParsedGame:
    """Validate and convert one game; raises ``GameRejected`` with a user-facing reason."""
    headers = game.headers
    variant = headers.get("Variant", "Standard").strip().lower()
    if variant not in SUPPORTED_VARIANTS:
        raise GameRejected(REASON_VARIANT, headers.get("Variant", ""))
    if raw is not None:
        tokens = mainline_tokens(raw)
        n_parsed = sum(1 for _ in game.mainline_moves())
        if len(tokens) > n_parsed:
            bad = _first_unreadable(tokens)
            raise GameRejected(
                REASON_UNREADABLE,
                f"read {n_parsed} of {len(tokens)} moves" + (f"; could not read '{bad}'" if bad else ""),
            )

    board = game.board()
    moves: list[ParsedMove] = []
    tc = headers.get("TimeControl")
    base_clock: float | None = None
    if tc and tc not in {"-", "?"} and "/" not in tc:
        try:
            base_clock = float(tc.split("+")[0])
        except ValueError:
            base_clock = None
    last_clock = {chess.WHITE: base_clock, chess.BLACK: base_clock}
    has_clock = False

    for ply, node in enumerate(game.mainline(), start=1):
        move = node.move
        if move is None or not board.is_legal(move):
            raise GameRejected(REASON_ILLEGAL, f"move {ply} is not legal in its position")
        mover = board.turn
        moves.append(
            ParsedMove(
                ply=ply,
                move_number=board.fullmove_number,
                color="white" if mover == chess.WHITE else "black",
                san=board.san(move),
                uci=move.uci(),
                fen_before=board.fen(),
                clock_before_s=last_clock[mover],
                features=position_features(board),
            )
        )
        clock = node.clock()
        if clock is not None:
            has_clock = True
            last_clock[mover] = float(clock)
        board.push(move)

    if not moves:
        raise GameRejected(REASON_EMPTY)
    if not has_clock:
        for m in moves:
            m.clock_before_s = None

    result = headers.get("Result", "*").strip()
    if result not in VALID_RESULTS:
        result = "*"
    eco = headers.get("ECO")
    eco = eco if eco and eco != "?" else None
    opening = headers.get("Opening") or opening_from_eco_url(headers.get("ECOUrl"))
    opening = opening if opening and opening != "?" else None
    played_at = parse_pgn_date(
        headers.get("UTCDate") or headers.get("Date"),
        headers.get("UTCTime") or headers.get("StartTime"),
    )

    exporter = chess.pgn.StringExporter(headers=True, variations=False, comments=False)
    return ParsedGame(
        white=clean_name(headers.get("White")),
        black=clean_name(headers.get("Black")),
        white_elo=parse_elo(headers.get("WhiteElo")),
        black_elo=parse_elo(headers.get("BlackElo")),
        result=result,
        played_at=played_at,
        time_control=tc if tc and tc != "?" else None,
        time_class=classify_time_control(tc),
        eco=eco,
        opening=opening,
        opening_family=opening_family(opening, eco),
        termination=headers.get("Termination"),
        site=headers.get("Site") or headers.get("Link"),
        source=source,
        moves=moves,
        pgn=game.accept(exporter),
    )


def parse_pgn_text(text: str, source: str = "pgn_upload", max_games: int | None = None) -> ParseReport:
    """Parse every game in ``text``. Malformed games are skipped and reported, never raised."""
    report = ParseReport()
    if not text or not text.strip():
        report.errors.append("The file is empty.")
        return report
    if "[" not in text and not re.search(r"\b1\.\s*\S", text):
        report.errors.append("No PGN content found (expected tag pairs like [White \"...\"] and moves).")
        return report

    handle = io.StringIO(text)
    index = 0
    seen: set[str] = set()
    while True:
        start = handle.tell()
        try:
            game = chess.pgn.read_game(handle)
        except Exception as exc:  # pragma: no cover - python-chess is very tolerant
            report.errors.append(f"Stopped reading after game {index}: {exc}")
            break
        if game is None:
            break
        raw = text[start:handle.tell()]
        if not raw.strip():
            break
        index += 1
        report.games_read += 1
        if game.errors:
            message = str(game.errors[0])
            reason = (REASON_ILLEGAL if "illegal" in message
                      else REASON_FEN if "fen" in message.lower() else REASON_UNREADABLE)
            report.reject(index, reason, str(game.errors[0]))
            continue
        try:
            parsed = _game_to_parsed(game, source, raw)
        except GameRejected as exc:
            report.reject(index, exc.reason, exc.detail)
            continue
        except Exception as exc:
            report.reject(index, REASON_ERROR, str(exc))
            continue
        if parsed.game_hash in seen:
            report.duplicates_in_file += 1
            continue
        seen.add(parsed.game_hash)
        report.games.append(parsed)
        if max_games is not None and len(report.games) >= max_games:
            report.truncated = bool(handle.read().strip())
            break

    if not report.games and not report.errors and not report.duplicates_in_file:
        report.errors.append("No complete games were found in the file.")
    return report


class UploadError(ValueError):
    """The uploaded bytes cannot be read as a PGN text file."""


def decode_pgn_bytes(raw: bytes) -> tuple[str, str | None]:
    """Decode an uploaded file. Returns ``(text, warning)``; raises ``UploadError``."""
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        try:
            return raw.decode("utf-16"), None
        except UnicodeDecodeError as exc:
            raise UploadError("The file claims to be UTF-16 but could not be decoded.") from exc
    try:
        return raw.decode("utf-8-sig"), None
    except UnicodeDecodeError:
        pass
    if b"\x00" in raw:
        raise UploadError("This does not look like a text PGN file (it contains binary data).")
    for encoding in ("cp1252", "latin-1"):
        try:
            text = raw.decode(encoding)
        except UnicodeDecodeError:
            continue
        return text, (
            "The file is not UTF-8, so it was read as Windows-1252 text. "
            "Accented player names may look wrong."
        )
    raise UploadError("The file's text encoding could not be recognized.")  # pragma: no cover


def decode_upload(raw: bytes) -> str:
    return decode_pgn_bytes(raw)[0]


def candidate_players(games: Iterable[ParsedGame]) -> list[tuple[str, int]]:
    """Players ordered by number of games - the most frequent is usually the uploader."""
    counts: Counter[str] = Counter()
    display: dict[str, str] = {}
    for g in games:
        for name in (g.white, g.black):
            if name == "Unknown":
                continue
            counts[name.lower()] += 1
            display.setdefault(name.lower(), name)
    return [(display[k], n) for k, n in counts.most_common()]


# ---------------------------------------------------------------------------
# Chess.com public API
# ---------------------------------------------------------------------------


class ChessComError(RuntimeError):
    """User-facing error from the Chess.com import."""


def _get_json(session, url: str, timeout: float, max_retries: int = 3, sleep=time.sleep) -> dict:
    """GET with timeout plus polite back-off on HTTP 429 (rate limit) and 5xx errors."""
    delay = 2.0
    for attempt in range(max_retries + 1):
        try:
            resp = session.get(
                url,
                timeout=timeout,
                headers={"User-Agent": config.CHESSCOM_USER_AGENT, "Accept": "application/json"},
            )
        except requests.Timeout as exc:
            raise ChessComError("Chess.com did not respond in time. Please try again shortly.") from exc
        except requests.RequestException as exc:
            raise ChessComError(f"Could not reach Chess.com ({exc.__class__.__name__}).") from exc

        if resp.status_code == 200:
            try:
                return resp.json()
            except ValueError as exc:
                raise ChessComError("Chess.com returned an unexpected response.") from exc
        if resp.status_code == 404:
            raise ChessComError("Chess.com player not found. Check the username spelling.")
        if resp.status_code in (429, 500, 502, 503, 504) and attempt < max_retries:
            retry_after = resp.headers.get("Retry-After") if hasattr(resp, "headers") else None
            wait = float(retry_after) if retry_after and str(retry_after).isdigit() else delay
            sleep(min(wait, 30.0))
            delay *= 2
            continue
        if resp.status_code == 429:
            raise ChessComError("Chess.com rate limit reached. Please wait a minute and try again.")
        raise ChessComError(f"Chess.com request failed (HTTP {resp.status_code}).")
    raise ChessComError("Chess.com request failed after retries.")  # pragma: no cover


def fetch_chesscom_games(
    username: str,
    max_games: int = config.CHESSCOM_DEFAULT_GAMES,
    session=None,
    progress: Callable[[str], None] | None = None,
    max_months: int = 24,
    sleep=time.sleep,
) -> ParseReport:
    """Download the player's most recent standard-chess games from monthly archives.

    Requests are made serially (Chess.com allows unlimited serial access but may
    rate-limit parallel requests).
    """
    username = (username or "").strip()
    if not USERNAME_RE.fullmatch(username):
        raise ChessComError("Enter a valid Chess.com username (letters, digits, '_' or '-').")
    max_games = max(1, min(int(max_games), config.CHESSCOM_MAX_GAMES))
    session = session or requests.Session()
    base = f"https://api.chess.com/pub/player/{username.lower()}/games/archives"

    archives = _get_json(session, base, config.CHESSCOM_TIMEOUT_S, sleep=sleep).get("archives", [])
    if not archives:
        raise ChessComError("This Chess.com account has no public games yet.")

    raw_games: list[dict] = []
    for month_url in list(reversed(archives))[:max_months]:
        if progress:
            progress(f"Downloading {month_url.rsplit('/games/', 1)[-1]} ...")
        payload = _get_json(session, month_url, config.CHESSCOM_TIMEOUT_S, sleep=sleep)
        month_games = [
            g for g in payload.get("games", []) if g.get("rules", "chess") == "chess" and g.get("pgn")
        ]
        month_games.sort(key=lambda g: g.get("end_time", 0), reverse=True)
        raw_games.extend(month_games)
        if len(raw_games) >= max_games:
            break

    raw_games = raw_games[:max_games]
    report = ParseReport()
    seen: set[str] = set()
    for g in reversed(raw_games):  # oldest first
        sub = parse_pgn_text(g["pgn"], source="chess.com")
        report.merge(sub)
        for parsed in sub.games:
            if parsed.game_hash in seen:
                report.duplicates_in_file += 1
                continue
            seen.add(parsed.game_hash)
            if g.get("time_class"):
                parsed.time_class = g["time_class"]
            if parsed.played_at is None and g.get("end_time"):
                parsed.played_at = datetime.fromtimestamp(int(g["end_time"]), tz=timezone.utc).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            if g.get("url"):
                parsed.site = g["url"]
            report.games.append(parsed)
    if not report.games:
        report.errors.append("No standard chess games were found in this account's recent archives.")
    return report
