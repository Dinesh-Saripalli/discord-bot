import os, csv, re, asyncio, datetime as dt, pathlib
from zoneinfo import ZoneInfo

import discord
from discord import app_commands
from dotenv import load_dotenv

# ---------------- Config / Env ----------------
load_dotenv()

TOKEN = os.getenv("DISCORD_TOKEN")
GUILD_ID = int(os.getenv("GUILD_ID"))
WORDLE_CHANNEL_ID = int(os.getenv("WORDLE_CHANNEL_ID"))
WORDLE_BOT_ID = int(os.getenv("WORDLE_BOT_ID", "0"))  # set to 0 to accept any bot author
CSV_PATH = os.getenv("CSV_PATH", "wordle_results.csv")
FLAG_PATH = pathlib.Path(os.getenv("BACKFILL_FLAG", "backfill_done.flag"))
TZ = ZoneInfo(os.getenv("WORDLE_TZ", "America/New_York"))

# --------------- Parsing patterns (more lenient) ---------------
# Flexible header: allow straight or curly apostrophe and any casing
HEADER_YDAY = re.compile(r"yesterday[’']?s results", re.IGNORECASE)

# Bucket lines like:
# "👑 2/6: @User", "**3/6:** @UserA @UserB", "X/6: @User"
# - optional crown
# - optional bold (** ... **)
# - tolerate extra spaces
BUCKET_LINE = re.compile(
    r"^\s*(?:👑\s*)?\**\s*([2-6X])\s*/\s*6\s*\**\s*:\s+(.+?)\s*$",
    re.IGNORECASE
)

# Mentions normally arrive as <@123...> in embed text. Keep this,
# but we'll add a fallback later for plain "@Name".
MENTION = re.compile(r"<@!?(\d+)>")

# --------------- CSV utilities ---------------
CSV_HEADERS = ["guild_id","user_id","result_date","attempts","won","source","created_at"]
file_lock = asyncio.Lock()
seen_keys = set()  # (guild_id, user_id, result_date)

def ensure_csv_exists():
    """Create CSV if needed and build the in-memory dedupe set."""
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(CSV_HEADERS)
        return
    # build dedupe index
    with open(CSV_PATH, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                key = (int(row["guild_id"]), int(row["user_id"]), row["result_date"])
                seen_keys.add(key)
            except Exception:
                continue

async def append_row_if_new(guild_id: int, user_id: int, result_date: str, attempts, won: bool, source: str):
    """Append if (guild_id,user_id,result_date) not yet recorded."""
    key = (guild_id, user_id, result_date)
    if key in seen_keys:
        return False
    now = dt.datetime.utcnow().isoformat()
    attempts_str = "" if attempts is None else str(int(attempts))
    won_str = "1" if won else "0"
    async with file_lock:
        with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([guild_id, user_id, result_date, attempts_str, won_str, source, now])
        seen_keys.add(key)
    return True

def read_all_rows():
    rows = []
    if not os.path.exists(CSV_PATH):
        return rows
    with open(CSV_PATH, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                rows.append({
                    "guild_id": int(r["guild_id"]),
                    "user_id": int(r["user_id"]),
                    "result_date": r["result_date"],
                    "attempts": (int(r["attempts"]) if r["attempts"] else None),
                    "won": True if r["won"] == "1" else False,
                    "source": r.get("source",""),
                    "created_at": r.get("created_at",""),
                })
            except Exception:
                continue
    return rows

# --------------- Discord setup ---------------
intents = discord.Intents.default()
intents.message_content = True   # keep on; safe in private server
intents.guilds = True

client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client)

@client.event
async def on_ready():
    ensure_csv_exists()
    await tree.sync(guild=discord.Object(id=GUILD_ID))
    print(f"Logged in as {client.user}")

    # Auto backfill once
    if not FLAG_PATH.exists():
        try:
            await run_backfill()
            FLAG_PATH.write_text("done")
            print("Backfill completed and flagged.")
        except Exception as e:
            print(f"Backfill error: {e}")

# --------------- Message filtering & parsing ---------------
def collect_embed_text(message: discord.Message) -> str:
    """Concatenate all embed text we care about."""
    blocks = []
    for e in message.embeds:
        if e.title:       blocks.append(e.title)
        if e.description: blocks.append(e.description)
        for f in e.fields:
            if f.value: blocks.append(f.value)
    return "\n".join(blocks)

def looks_like_recap(text: str) -> bool:
    """True if we see the 'yesterday’s results' header OR at least one bucket line."""
    if HEADER_YDAY.search(text):
        return True
    for line in text.splitlines():
        if BUCKET_LINE.match(line.strip()):
            return True
    return False

def is_wordle_recap_message(message: discord.Message) -> bool:
    if message.guild is None or message.channel.id != WORDLE_CHANNEL_ID:
        return False
    if not message.author.bot:
        return False
    if WORDLE_BOT_ID and message.author.id != WORDLE_BOT_ID:
        return False
    if not message.embeds:
        return False
    text = collect_embed_text(message)
    return looks_like_recap(text)

def parse_recap_embed(message: discord.Message):
    """
    Returns list of (user_id, attempts:int|None, won:bool) from lines like:
      👑 2/6: @UserA
      **3/6:** @UserB @UserC
      X/6: @UserD
    """
    results = []
    text = collect_embed_text(message)

    # Pass 1: parse ID-based mentions directly found in the line
    lines = [ln.strip() for ln in text.splitlines()]
    for line in lines:
        m = BUCKET_LINE.match(line)
        if not m:
            continue
        token = m.group(1).upper()
        user_segment = m.group(2)
        ids = [int(s) for s in MENTION.findall(user_segment)]

        won = token != "X"
        attempts = None if token == "X" else int(token)

        # If we got IDs from the text, great — use them.
        if ids:
            for uid in ids:
                results.append((uid, attempts, won))
            continue

        # Fallback: resolve plain @DisplayName tokens (rare, but just in case)
        # This tries exact matches against member.display_name or member.global_name.
        # Note: Names with spaces/emoji may still fail; this is only a fallback.
        tokens = [t for t in user_segment.split() if t.startswith("@")]
        if tokens:
            guild = message.guild
            members = {m.id: m for m in guild.members}  # cache current members
            lower_map = {}
            for mbr in members.values():
                if mbr.display_name:
                    lower_map["@"+mbr.display_name.lower()] = mbr.id
                if getattr(mbr, "global_name", None):
                    lower_map["@"+mbr.global_name.lower()] = mbr.id

            for t in tokens:
                uid = lower_map.get(t.lower())
                if uid:
                    results.append((uid, attempts, won))
                # else: skip unknown token

    return results

def recap_result_date(message: discord.Message) -> str:
    """Use message timestamp in TZ to compute 'yesterday' (the recap's date)."""
    local = message.created_at.replace(tzinfo=dt.timezone.utc).astimezone(TZ)
    return (local - dt.timedelta(days=1)).date().isoformat()

# --------------- Live ingest ---------------
@client.event
async def on_message(message: discord.Message):
    if not is_wordle_recap_message(message):
        return
    parsed = parse_recap_embed(message)
    if not parsed:
        return
    rdate = recap_result_date(message)
    for uid, attempts, won in parsed:
        await append_row_if_new(message.guild.id, uid, rdate, attempts, won, "summary_embed")

# --------------- Backfill (one-time full history scan) ---------------
async def run_backfill():
    """Walk the entire channel history and log all recap results."""
    guild = client.get_guild(GUILD_ID) or await client.fetch_guild(GUILD_ID)
    channel = guild.get_channel(WORDLE_CHANNEL_ID) or await client.fetch_channel(WORDLE_CHANNEL_ID)

    print("Starting backfill… (scanning full channel history)")
    count_msgs = 0
    count_add = 0

    async for msg in channel.history(limit=None, oldest_first=True):
        count_msgs += 1
        if not is_wordle_recap_message(msg):
            continue
        parsed = parse_recap_embed(msg)
        if not parsed:
            continue
        rdate = recap_result_date(msg)
        for uid, attempts, won in parsed:
            added = await append_row_if_new(msg.guild.id, uid, rdate, attempts, won, "summary_embed")
            if added:
                count_add += 1
        if count_msgs % 500 == 0:
            print(f"Backfill progress: scanned {count_msgs} messages, wrote {count_add} rows…")

    print(f"Backfill complete: scanned {count_msgs} messages, wrote {count_add} new rows.")

# --------------- Slash commands ---------------
@tree.command(name="leaderboard", description="Wordle leaderboard", guild=discord.Object(id=GUILD_ID))
async def leaderboard(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=True, thinking=True)
    rows = [r for r in read_all_rows() if r["guild_id"] == interaction.guild_id]
    if not rows:
        await interaction.followup.send("No data yet — wait for a daily recap.")
        return

    # aggregate
    stats = {}  # user_id -> dict
    for r in rows:
        u = stats.setdefault(r["user_id"], {"games":0,"wins":0,"fails":0,"sum_attempts":0,"win_samples":0})
        u["games"] += 1
        if r["won"]:
            u["wins"] += 1
            if r["attempts"] is not None:
                u["sum_attempts"] += r["attempts"]
                u["win_samples"] += 1
        else:
            u["fails"] += 1

    items = []
    for uid, u in stats.items():
        win_rate = (100.0*u["wins"]/u["games"]) if u["games"] else 0.0
        avg_att = (u["sum_attempts"]/u["win_samples"]) if u["win_samples"] else None
        items.append((uid, u["games"], u["wins"], u["fails"], win_rate, avg_att))

    # sort: win_rate desc, avg_attempts asc (None last), games desc
    def sort_key(t):
        uid, games, wins, fails, win_rate, avg_att = t
        return (-win_rate, (avg_att if avg_att is not None else 1e9), -games)
    items.sort(key=sort_key)

    lines = ["**Wordle Leaderboard**"]
    for i, (uid, games, wins, fails, win_rate, avg_att) in enumerate(items, start=1):
        member = interaction.guild.get_member(uid) or await interaction.guild.fetch_member(uid)
        name = member.display_name if member else f"<@{uid}>"
        avg_txt = f"{avg_att:.2f}" if avg_att is not None else "—"
        lines.append(f"`#{i}` **{name}** — {win_rate:.1f}% win • {wins}/{games} • avg {avg_txt} • fails {fails}")

    await interaction.followup.send("\n".join(lines))

@tree.command(name="report", description="Upload the CSV report", guild=discord.Object(id=GUILD_ID))
async def report(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=True, thinking=True)
    async with file_lock:
        pass  # ensure no concurrent write
    if not os.path.exists(CSV_PATH):
        await interaction.followup.send("No CSV yet.")
        return
    await interaction.followup.send(file=discord.File(CSV_PATH, filename=os.path.basename(CSV_PATH)), ephemeral=True)

@tree.command(name="backfill", description="Scan all past recap messages (one-time).", guild=discord.Object(id=GUILD_ID))
async def backfill_cmd(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=True, thinking=True)
    if FLAG_PATH.exists():
        await interaction.followup.send("Backfill already completed. Delete `backfill_done.flag` to run again.")
        return
    try:
        await run_backfill()
        FLAG_PATH.write_text("done")
        await interaction.followup.send("Backfill completed ✅")
    except Exception as e:
        await interaction.followup.send(f"Backfill failed: `{e}`")

# --------------- Entry point ---------------
if __name__ == "__main__":
    if not TOKEN:
        raise SystemExit("Missing DISCORD_TOKEN in .env")
    client.run(TOKEN)
