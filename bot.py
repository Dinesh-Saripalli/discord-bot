# bot.py
# Discord Wordle tracker with forensic debug utilities
# Handles content-only Wordle recaps (no embeds) and plain @name tokens.
# Requires: discord.py, python-dotenv  (pip install -r requirements.txt)

import os
import re
import csv
import sys
import asyncio
import logging
import datetime as dt
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple
from discord import app_commands
import discord
from dotenv import load_dotenv

# -----------------------------------------------------------------------------
# Load env & set aliases for legacy names you already use
# -----------------------------------------------------------------------------

load_dotenv()  # load .env alongside this script or current working directory

# Accept legacy env names so your existing .env works:
if not os.environ.get("WORDLE_APP_ID") and os.environ.get("WORDLE_BOT_ID"):
    os.environ["WORDLE_APP_ID"] = os.environ["WORDLE_BOT_ID"]

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
CWD = Path.cwd()

TOKEN = os.environ.get("DISCORD_TOKEN") or os.environ.get("TOKEN")
if not TOKEN:
    print("[fatal] DISCORD_TOKEN (or TOKEN) is missing in .env")
    sys.exit(1)

def _env_int(name: str, default: Optional[int] = None) -> int:
    v = (os.environ.get(name) or "").strip()
    if v.isdigit():
        return int(v)
    if default is not None:
        return default
    raise RuntimeError(f"Missing or invalid env: {name}")

GUILD_ID              = _env_int("GUILD_ID")
WORDLE_CHANNEL_ID     = _env_int("WORDLE_CHANNEL_ID")
# 0 means "not configured" — we'll try to auto-learn it on first valid Wordle recap.
WORDLE_APP_ID         = _env_int("WORDLE_APP_ID", 0)

CSV_PATH = os.environ.get("CSV_PATH") or str(SCRIPT_DIR / "wordle_results.csv")
CSV_PATH = str(Path(CSV_PATH))

WORDLE_BACKFILL_START = os.environ.get("WORDLE_BACKFILL_START", "2025-09-11")

# Forensic debug controls
DEBUG_MESSAGE_ID_STR  = os.environ.get("DEBUG_MESSAGE_ID", "").strip()
DEBUG_MESSAGE_ID      = int(DEBUG_MESSAGE_ID_STR) if DEBUG_MESSAGE_ID_STR.isdigit() else None
DEBUG_SCAN_LIMIT      = int(os.environ.get("DEBUG_SCAN_LIMIT", "300"))  # how many latest msgs to sample for look-alikes

# Optional timezone for interpreting WORDLE_BACKFILL_START local midnight & puzzle derivation
TZ_NAME = os.environ.get("WORDLE_TZ", "America/New_York").strip()

try:
    from zoneinfo import ZoneInfo  # Py3.9+
    ZONE = ZoneInfo(TZ_NAME) if TZ_NAME else None
except Exception:
    ZONE = None  # Fallback to naive/UTC handling if tz not available

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("wordlebot")

print(f"[startup] cwd={CWD}")
print(f"[startup] script_dir={SCRIPT_DIR}")
print(f"[startup] CSV_PATH={CSV_PATH}")
print(
    f"[startup] GUILD_ID={GUILD_ID} WORDLE_CHANNEL_ID={WORDLE_CHANNEL_ID} "
    f"WORDLE_APP_ID={WORDLE_APP_ID} WORDLE_BOT_ID={os.environ.get('WORDLE_BOT_ID','<unset>')}"
)
print(f"[startup] WORDLE_BACKFILL_START (local TZ) = {WORDLE_BACKFILL_START}")
if TZ_NAME:
    print(f"[startup] WORDLE_TZ={TZ_NAME}")
if DEBUG_MESSAGE_ID:
    print(f"[startup] DEBUG_MESSAGE_ID={DEBUG_MESSAGE_ID}")
print(f"[startup] DEBUG_SCAN_LIMIT={DEBUG_SCAN_LIMIT}")

# -----------------------------------------------------------------------------
# Discord client & intents
# -----------------------------------------------------------------------------

intents = discord.Intents.default()
intents.guilds = True
intents.messages = True         # receive guild message events + history()
intents.message_content = True  # required for content-only recaps
client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client)


# -----------------------------------------------------------------------------
# Parsing utilities (recap posts from the Wordle App)
# -----------------------------------------------------------------------------

# Example embed title: "Wordle No. 1566" (some servers have this; yours doesn't for recaps)
EMBED_TITLE_RE = re.compile(r"Wordle\s+No\.\s*(\d+)", re.IGNORECASE)

# Recap lines in content/description/fields:
#   "👑 2/6: <@111>"
#   "3/6: @Starbreaker <@2604> <@2673> ..."
LINE_RECAP_RE  = re.compile(r"(?m)^[^\S\r\n]*(?:👑\s*)?([1-6Xx])\/6:\s*(.+?)\s*$")

# Extract actual mention ids in Discord content (<@12345> or <@!12345>)
MENTION_ID_RE  = re.compile(r"<@!?(\d+)>")

# Extract plain @name tokens (non-mention), conservative to avoid emails/URLs
PLAIN_AT_RE    = re.compile(r"(?<!<)@([A-Za-z0-9_.\-]+)")

CSV_FIELDS = [
    "timestamp_utc",
    "puzzle",
    "score",
    "author_id",
    "author_name",
    "message_id",
]

# Wordle epoch: Wordle #0 on 2021-06-19
WORDLE_EPOCH = dt.date(2021, 6, 19)

def to_utc(d: dt.datetime) -> dt.datetime:
    """Ensure a datetime is timezone-aware UTC."""
    if d.tzinfo is None:
        return d.replace(tzinfo=dt.timezone.utc)
    return d.astimezone(dt.timezone.utc)

def start_after_utc(local_date_str: str) -> dt.datetime:
    """
    Interpret WORDLE_BACKFILL_START as midnight in WORDLE_TZ (if provided), then convert to UTC.
    If WORDLE_TZ is not set/available, treat it as naive local → attach UTC.
    """
    date_obj = dt.datetime.strptime(local_date_str, "%Y-%m-%d").date()
    if ZONE:
        local_midnight = dt.datetime.combine(date_obj, dt.time(0, 0, 0), tzinfo=ZONE)
        return local_midnight.astimezone(dt.timezone.utc)
    return to_utc(dt.datetime.combine(date_obj, dt.time(0, 0, 0)))

def get_puzzle_from_embeds(msg: discord.Message) -> Optional[int]:
    for e in msg.embeds or []:
        if e.title:
            m = EMBED_TITLE_RE.search(e.title)
            if m:
                try:
                    return int(m.group(1))
                except ValueError:
                    pass
    return None

def derive_puzzle_from_created_at(msg: discord.Message) -> Optional[int]:
    """
    Many recaps say 'yesterday's results' and are posted after midnight local time.
    We convert created_at to WORDLE_TZ and subtract 1 day to get the puzzle date,
    then map date -> Wordle number using the Wordle epoch.
    """
    if not msg.created_at:
        return None
    # Convert created_at (UTC) → local tz
    local_dt = msg.created_at.astimezone(ZONE) if ZONE else msg.created_at
    puzzle_date = (local_dt - dt.timedelta(days=1)).date()
    # Derive number
    days = (puzzle_date - WORDLE_EPOCH).days
    return days if days >= 0 else None

def extract_recap_text(msg: discord.Message) -> str:
    """
    Combine message.content, embed.description, and embed.fields[*].value so we can parse recaps
    even if message content intent is disabled or Wordle moves text into fields.
    """
    parts: List[str] = []
    if msg.content:
        parts.append(msg.content)

    for e in (msg.embeds or []):
        if getattr(e, "description", None):
            parts.append(e.description)
        if getattr(e, "fields", None):
            for fld in e.fields:
                if getattr(fld, "value", None):
                    parts.append(str(fld.value))

    return "\n".join(p for p in parts if p).strip()

def parse_wordle_recap(msg: discord.Message) -> List[Dict[str, str]]:
    """
    From a Wordle App recap message, produce one row per player.
    Works with content-only recaps and with embed-title recaps.
    1) If an explicit puzzle number exists in embeds, use it.
    2) Otherwise derive puzzle from created_at & WORDLE_TZ (yesterday).
    3) Capture both true mentions (<@id>) and plain @name tokens.
    """
    # Auto-learning: If this looks like a recap and WORDLE_APP_ID is unknown, learn it.
    global WORDLE_APP_ID

    text = extract_recap_text(msg)
    if not text:
        return []

    # Identify recap lines first; if no recap lines, bail
    if not LINE_RECAP_RE.search(text):
        return []

    # Enforce app id if configured; if not configured and this looks like a recap, learn it
    if WORDLE_APP_ID and WORDLE_APP_ID != 0:
        if msg.author.id != WORDLE_APP_ID:
            return []
    else:
        WORDLE_APP_ID = msg.author.id
        print(f"[learned] WORDLE_APP_ID set to {WORDLE_APP_ID} from message {msg.id}")

    # Get puzzle number (prefer embeds, else derive from created_at)
    puzzle = get_puzzle_from_embeds(msg)
    if puzzle is None:
        puzzle = derive_puzzle_from_created_at(msg)

    # If still None, we’ll store as -1 to indicate unknown but keep the data
    if puzzle is None:
        puzzle = -1

    rows: List[Dict[str, str]] = []
    id_to_name: Dict[int, str] = {u.id: f"{u.name}#{u.discriminator}" for u in (msg.mentions or [])}

    for m in LINE_RECAP_RE.finditer(text):
        score = m.group(1).upper()  # '1'..'6' or 'X'
        tail  = m.group(2)

        # True mentions
        ids   = [int(x) for x in MENTION_ID_RE.findall(tail)]
        for uid in ids:
            rows.append({
                "timestamp_utc": to_utc(msg.created_at).isoformat(),
                "puzzle": puzzle,
                "score": score,
                "author_id": str(uid),
                "author_name": id_to_name.get(uid, ""),
                "message_id": str(msg.id),
            })


        # Plain @name tokens (exclude ones that were actual mentions)
        # Note: this is best-effort; you can later map names → IDs.
        # Plain @name tokens (non-mentions). We always record them; they dedupe via (message_id + author_name).
        plain_names: List[str] = PLAIN_AT_RE.findall(tail)
        for nm in plain_names:
            rows.append({
                "timestamp_utc": to_utc(msg.created_at).isoformat(),
                "puzzle": puzzle,
                "score": score,
                "author_id": "",            # unknown for plain @names
                "author_name": f"@{nm}",    # keep the token so it’s usable in stats
                "message_id": str(msg.id),
            })


    return rows

# -----------------------------------------------------------------------------
# CSV helpers & duplicate suppression
# -----------------------------------------------------------------------------

def ensure_csv(csv_path: str):
    p = Path(csv_path)
    needs_header = (not p.exists()) or (p.stat().st_size == 0)
    f = open(p, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
    if needs_header:
        w.writeheader()
    return f, w

def load_existing_keys(csv_path: str) -> Set[str]:
    """
    Return set of '(message_id):(author_id or @name)' keys for fast duplicate suppression.
    """
    keys: Set[str] = set()
    p = Path(csv_path)
    if not (p.exists() and p.stat().st_size > 0):
        return keys
    with open(p, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            aid = row.get("author_id", "")
            anm = row.get("author_name", "")
            unique_author = aid if aid else anm  # use @name when id is empty
            k = f'{row.get("message_id","")}:{unique_author}'
            if k != ":":
                keys.add(k)
    return keys

# -----------------------------------------------------------------------------
# Forensic debug: dump a message and scan for look-alikes
# -----------------------------------------------------------------------------

def _s(val):
    """Safe short string for printing (avoid huge lines)."""
    if val is None:
        return None
    s = str(val)
    return s if len(s) <= 300 else s[:300] + " …(truncated)"

def summarize_embed(e: discord.Embed) -> Dict[str, Optional[str]]:
    fields = []
    for fld in (e.fields or []):
        fields.append({
            "name": _s(fld.name),
            "value": _s(fld.value),
            "inline": getattr(fld, "inline", None),
        })
    return {
        "title": _s(e.title),
        "description": _s(e.description),
        "url": e.url,
        "type": getattr(e, "type", None),
        "footer": _s(getattr(e.footer, "text", None)) if getattr(e, "footer", None) else None,
        "author": _s(getattr(e.author, "name", None)) if getattr(e, "author", None) else None,
        "fields": fields,
    }

async def dump_message_details(ch: discord.TextChannel, mid: int):
    try:
        msg = await ch.fetch_message(mid)
    except discord.NotFound:
        print(f"[debug] Message {mid} not found in this channel.")
        return
    except discord.Forbidden:
        print(f"[debug] Forbidden fetching message {mid} (check permissions).")
        return
    except Exception as e:
        print(f"[debug] Error fetching message {mid}: {e}")
        return

    print("\n========== DEBUG DUMP (single message) ==========")
    print(f"id={msg.id} jump_url={msg.jump_url}")
    print(f"author={msg.author} id={msg.author.id} bot={getattr(msg.author, 'bot', False)}")
    print(f"created_at={msg.created_at} edited_at={msg.edited_at}")
    print(f"pinned={msg.pinned} tts={msg.tts} type={msg.type}")
    print(f"content: {_s(msg.content)}")
    print(f"attachments: {[a.filename for a in msg.attachments]}")
    print(f"mentions(user ids): {[u.id for u in msg.mentions]}")
    print(f"role_mentions: {[r.id for r in msg.role_mentions]}")
    print(f"channel_mentions: {[c.id for c in msg.channel_mentions]}")
    print(f"reference: {msg.reference}")
    print(f"components: {msg.components}")

    # Embeds
    if msg.embeds:
        print(f"embeds_count={len(msg.embeds)}")
        for i, e in enumerate(msg.embeds):
            summ = summarize_embed(e)
            print(f"[embed {i}] {summ}")
    else:
        print("embeds_count=0")

    # Parser perspective
    puzzle_from_embeds = get_puzzle_from_embeds(msg)
    puzzle_derived = derive_puzzle_from_created_at(msg)
    text = extract_recap_text(msg)
    rows = parse_wordle_recap(msg)

    print(f"puzzle_from_embeds={puzzle_from_embeds}  puzzle_derived={puzzle_derived}")
    print(f"extract_recap_text (first 300): {_s(text)}")
    print(f"parse_wordle_recap -> rows={len(rows)}")
    for i, r in enumerate(rows[:10]):
        print(f"  row[{i}] {r}")
    if len(rows) > 10:
        print(f"  ... plus {len(rows)-10} more rows")
    print("========== END DEBUG DUMP ==========\n")

def classify_skip_reason(msg: discord.Message) -> str:
    """Return a short reason if we would skip this message during parsing."""
    text = extract_recap_text(msg)
    if not text or not LINE_RECAP_RE.search(text):
        return "skip:not_recap_lines"
    if WORDLE_APP_ID and WORDLE_APP_ID != 0 and msg.author.id != WORDLE_APP_ID:
        return f"skip:author!=WORDLE_APP_ID({WORDLE_APP_ID})"
    # Mentions or plain @names must exist
    if not (MENTION_ID_RE.search(text) or PLAIN_AT_RE.search(text)):
        return "skip:no_mentions_or_plain_names"
    return "ok"

async def scan_lookalikes(ch: discord.TextChannel, limit: int = 300):
    """
    Scan the latest `limit` messages and summarize which ones look like
    Wordle recaps (by recap lines), plus the skip reasons for those we ignore.
    """
    print(f"\n========== DEBUG SCAN (last {limit} messages) ==========")
    count = 0
    wordleish = 0
    ok = 0
    by_reason: Dict[str, int] = {}
    async for msg in ch.history(limit=limit, oldest_first=False):
        count += 1
        text = extract_recap_text(msg)
        if not text:
            continue
        if not LINE_RECAP_RE.search(text):
            continue
        wordleish += 1
        reason = classify_skip_reason(msg)
        by_reason[reason] = by_reason.get(reason, 0) + 1
        if reason == "ok":
            ok += 1

    print(f"scanned={count}  wordleish(has recap lines)={wordleish}  ok={ok}")
    for reason, n in sorted(by_reason.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {reason:>34}  : {n}")
    print("========== END DEBUG SCAN ==========\n")

    # ===== simple CSV reader (used by catchup) =====
def _load_rows(csv_path: str) -> List[Dict[str, str]]:
    """
    Load all rows from the CSV (returns [] if file doesn't exist/empty).
    """
    p = Path(csv_path)
    if not (p.exists() and p.stat().st_size > 0):
        return []
    with p.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return list(r)


# -----------------------------------------------------------------------------
# Backfill & incremental
# -----------------------------------------------------------------------------

def unique_key_for_row(r: Dict[str, str]) -> str:
    # Dedup by message_id + (author_id or @name token)
    unique_author = r["author_id"] if r["author_id"] else r["author_name"]
    return f'{r["message_id"]}:{unique_author}'

from collections import defaultdict

def _load_rows(csv_path: str) -> List[Dict[str, str]]:
    p = Path(csv_path)
    if not (p.exists() and p.stat().st_size > 0):
        return []
    with p.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return list(r)

def _is_win(score: str) -> bool:
    s = (score or "").upper().strip()
    return s in {"1","2","3","4","5","6"}

def _score_val(score: str) -> Optional[int]:
    s = (score or "").upper().strip()
    if s.isdigit():
        return int(s)
    return None  # X/unknown

def compute_leaderboard(rows: List[Dict[str, str]], window_days: Optional[int] = None):
    cutoff = None
    if window_days:
        cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=window_days)

    by_user = defaultdict(lambda: {
        "name": "",
        "games": 0, "wins": 0, "losses": 0,
        "guess_sum": 0, "guess_count": 0,
        "dist": defaultdict(int),
        "puzzles": set(), "by_puzzle": {},
    })

    for r in rows:
        try:
            ts_dt = dt.datetime.fromisoformat(r.get("timestamp_utc",""))
        except Exception:
            ts_dt = None
        if cutoff and ts_dt and ts_dt < cutoff:
            continue

        try:
            puzzle = int(r.get("puzzle", -1))
        except Exception:
            puzzle = -1

        score = (r.get("score") or "").upper().strip()
        aid = (r.get("author_id") or "").strip()
        aname = (r.get("author_name") or "").strip()
        key = aid if aid else aname
        if not key:
            continue

        u = by_user[key]
        u["name"] = aname if aname else key
        u["games"] += 1
        if _is_win(score):
            u["wins"] += 1
            sv = _score_val(score)
            if sv is not None:
                u["guess_sum"] += sv
                u["guess_count"] += 1
        else:
            u["losses"] += 1

        u["dist"][score if score else "X"] += 1
        if puzzle >= 0:
            u["puzzles"].add(puzzle)
            u["by_puzzle"][puzzle] = score

    def streaks(u):
        if not u["puzzles"]:
            return 0,0
        seq = sorted(u["puzzles"])
        # max streak forward
        last = None; run = 0; mx = 0
        for p in seq:
            s = u["by_puzzle"][p]
            if last is None or p == last + 1:
                run = run + 1 if _is_win(s) else 0
            else:
                run = 1 if _is_win(s) else 0
            mx = max(mx, run); last = p
        # current streak backward
        cur = 0; lastp = None
        for p in reversed(seq):
            s = u["by_puzzle"][p]
            if lastp is None:
                if _is_win(s):
                    cur = 1; lastp = p
                else:
                    break
            else:
                if p == lastp - 1 and _is_win(s):
                    cur += 1; lastp = p
                else:
                    break
        return cur, mx

    stats = {}
    for k,u in by_user.items():
        cur,mx = streaks(u)
        avg = (u["guess_sum"]/u["guess_count"]) if u["guess_count"] else None
        wr = (u["wins"]/u["games"]*100) if u["games"] else 0.0
        dist = {d: u["dist"].get(d,0) for d in ["1","2","3","4","5","6","X"]}
        stats[k] = {
            "name": u["name"], "games": u["games"], "wins": u["wins"], "losses": u["losses"],
            "avg": avg, "winrate": wr, "cur_streak": cur, "max_streak": mx, "dist": dist
        }

    ranking = sorted(
        stats.keys(),
        key=lambda k: (-stats[k]["wins"], -stats[k]["winrate"], stats[k]["avg"] if stats[k]["avg"] is not None else 999, -stats[k]["games"])
    )
    return stats, ranking
@tree.command(
    name="leaderboard",
    description="Show the Wordle leaderboard (optionally last N days)",
    guild=discord.Object(id=GUILD_ID),  # guild-scoped so it's available immediately
)
@app_commands.describe(window_days="Only include games from the last N days")
async def leaderboard_slash(
    interaction: discord.Interaction,
    window_days: Optional[int] = None
):
    # acknowledge quickly so Discord doesn't time out
    await interaction.response.defer(ephemeral=False)

    rows = _load_rows(CSV_PATH)
    if not rows:
        await interaction.followup.send("No data yet. Wait for a recap or run a catch-up.")
        return

    stats, ranking = compute_leaderboard(rows, window_days=window_days)
    top = ranking[:10]
    if not top:
        await interaction.followup.send("No results to show.")
        return

    header = f"**Wordle Leaderboard**{' — last ' + str(window_days) + 'd' if window_days else ''}"
    lines = [header]
    for i, key in enumerate(top, 1):
        s = stats[key]
        avg = f"{s['avg']:.2f}" if s["avg"] is not None else "–"
        lines.append(
            f"**{i}. {s['name']}** — Wins: {s['wins']}/{s['games']} "
            f"(WR {s['winrate']:.0f}%), Avg: {avg}, Streak: {s['cur_streak']} (max {s['max_streak']})"
        )

    out = "\n".join(lines)
    if len(out) <= 2000:
        await interaction.followup.send(out)
    else:
        # split safely if the message is too long
        for i in range(0, len(out), 1800):
            await interaction.followup.send(out[i:i+1800])


async def catchup_since_last_csv(buffer_days: int = 2):
    """
    On startup, catch up any missed recaps by scanning history after the last
    timestamp we have in the CSV (minus a small buffer). No flag file needed.
    """
    guild = client.get_guild(GUILD_ID)
    if not guild:
        log.error("Guild not found: %s", GUILD_ID)
        return
    ch = guild.get_channel(WORDLE_CHANNEL_ID)
    if not isinstance(ch, discord.TextChannel):
        log.error("Channel not found or not a TextChannel: %s", WORDLE_CHANNEL_ID)
        return

    # figure out where to start
    rows = _load_rows(CSV_PATH)
    last_ts: dt.datetime | None = None
    for r in rows:
        try:
            ts = dt.datetime.fromisoformat(r.get("timestamp_utc",""))
            if (last_ts is None) or (ts > last_ts):
                last_ts = ts
        except Exception:
            pass

    if last_ts is None:
        # Fall back to your configured start date (first run scenarios)
        after_utc = start_after_utc(WORDLE_BACKFILL_START)
    else:
        # small overlap buffer for safety
        after_utc = last_ts - dt.timedelta(days=buffer_days)
        if after_utc.tzinfo is None:
            after_utc = after_utc.replace(tzinfo=dt.timezone.utc)

    existing = load_existing_keys(CSV_PATH)
    wrote = 0
    scanned = 0
    recap_msgs = 0

    print(f"[catchup] starting from {after_utc.isoformat()} (buffer={buffer_days}d)")
    f, w = ensure_csv(CSV_PATH)
    with f:
        async for msg in ch.history(limit=None, after=after_utc, oldest_first=True):
            scanned += 1
            rows_new = parse_wordle_recap(msg)
            if not rows_new:
                continue
            recap_msgs += 1
            for r in rows_new:
                k = unique_key_for_row(r)
                if k in existing:
                    continue
                w.writerow(r)
                existing.add(k)
                wrote += 1
    print(f"[catchup] scanned={scanned}, recaps={recap_msgs}, wrote={wrote}")

@client.event
async def on_ready():
    print(f"Logged in as {client.user}")

    # Ensure CSV exists
    f, _ = ensure_csv(CSV_PATH); f.close()

    # Sync app commands to this guild (fast; avoids global propagation delay)
    try:
        await tree.sync(guild=discord.Object(id=GUILD_ID))
        print("[startup] slash commands synced to guild")
    except Exception as e:
        print(f"[startup] slash sync failed: {e}")

    # Optional forensic scan
    guild = client.get_guild(GUILD_ID)
    ch = guild.get_channel(WORDLE_CHANNEL_ID) if guild else None
    if isinstance(ch, discord.TextChannel) and DEBUG_SCAN_LIMIT > 0:
        await scan_lookalikes(ch, DEBUG_SCAN_LIMIT)

    # Catch up missed messages
    asyncio.create_task(catchup_since_last_csv(buffer_days=2))


@client.event
async def on_message(msg: discord.Message):
    """
    Incremental updates: whenever a new Wordle App recap appears in the configured channel,
    append rows for all players mentioned/plain-named in the recap.
    """
    if msg.guild is None:
        return
    if msg.channel.id != WORDLE_CHANNEL_ID:
        return

    rows = parse_wordle_recap(msg)
    if not rows:
        return

    existing = load_existing_keys(CSV_PATH)
    f, w = ensure_csv(CSV_PATH)
    wrote = 0
    with f:
        for r in rows:
            k = unique_key_for_row(r)
            if k in existing:
                continue
            w.writerow(r)
            existing.add(k)
            wrote += 1

    if wrote:
        log.info("Appended %d new row(s) from recap %s", wrote, msg.id)

# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        client.run(TOKEN, log_handler=None)
    except KeyboardInterrupt:
        print("Shutting down…")
