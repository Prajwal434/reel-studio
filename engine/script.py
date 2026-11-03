"""Screenplay / script support.
parse():        "NAME: line" / "NAME (V.O.): line" / NAME on its own line followed by dialogue -> [(speaker, text)]
ai_lines():     the AI character's lines, in order (fills AI-skit reply slots)
fix_captions(): replace mis-heard caption chunks with the script's words (fuzzy alignment of each chunk to the
                creator's lines), keeping the caption timing."""
import re
from difflib import SequenceMatcher

AI_NAMES = {"ai", "chatgpt", "gpt", "assistant", "bot", "claude", "gemini", "siri", "alexa", "laptop", "computer"}
STAGE = re.compile(r"^\s*(\(.*\)|\[.*\]|scene\b.*|shot\b.*|editing\b.*|int\.|ext\.|duration\b.*|format\b.*|style\b.*|on-screen text\b.*)\s*$", re.I)


def parse(text):
    out, speaker = [], None
    for raw in (text or "").splitlines():
        line = raw.strip().strip("*").strip()
        if not line or STAGE.match(line):
            speaker = None if not line else speaker; continue
        m = re.match(r"^([A-Za-z][\w .'\-]{0,30}?)\s*(\([^)]*\))?\s*[:：]\s*(.+)$", line)
        if m:
            speaker, said = m.group(1).strip(), m.group(3).strip()
            out.append((speaker, said)); continue
        if re.fullmatch(r"[A-Z][A-Z0-9 .'\-]{1,30}(\s*\([^)]*\))?", line):  # screenplay style: NAME on its own line
            speaker = re.sub(r"\s*\(.*\)$", "", line).strip(); continue
        if speaker:
            if out and out[-1][0] == speaker: out[-1] = (speaker, out[-1][1] + " " + line)
            else: out.append((speaker, line))
    return [(sp, re.sub(r"\s*\([^)]*\)\s*", " ", t).strip().strip('"“”')) for sp, t in out if t.strip()]


def is_ai(speaker, ai_name=""):
    s = speaker.lower().replace(" ", "")
    return s in AI_NAMES or (ai_name and s == ai_name.lower().replace(" ", "")) or "gpt" in s


def ai_lines(text, ai_name=""):
    return [t for sp, t in parse(text) if is_ai(sp, ai_name)]


def creator_text(text, ai_name=""):
    lines = parse(text)
    if not lines:  # plain prose / lyrics: the whole text is what the creator says
        return " ".join(l.strip() for l in (text or "").splitlines() if l.strip() and not STAGE.match(l))
    return " ".join(t for sp, t in lines if not is_ai(sp, ai_name))


norm = lambda w: re.sub(r"[^a-z0-9']", "", w.lower())


def fix_captions(captions, text, ai_name="", case=None):
    """fix mis-heard words only. Each caption chunk is matched (letter by letter) to the closest stretch of the creator's
    script; then a heard word is swapped for a script word only when it's clearly a mis-hearing of it ("billing" ->
    "building", "off-page" -> "off-pitch"). Words he improvised (not in the script) stay: captions show what was said."""
    words = creator_text(text, ai_name).split()
    if not words: return captions, 0
    keys = [norm(w) for w in words]
    sim = lambda a, b: SequenceMatcher(None, a, b).ratio()
    fixed, pos = 0, 0
    for c in captions:
        cws = c["text"].split()
        joined = "".join(norm(w) for w in cws)
        if len(joined) < 3: continue
        n = len(cws); best = (0.0, None, None)
        for start in range(len(keys)):
            for L in range(max(1, n - 2), n + 3):
                if start + L > len(keys): break
                r = sim(joined, "".join(keys[start:start + L])) - 0.002 * abs(start - pos)  # prefer the next lines
                if r > best[0]: best = (r, start, L)
        r, start, L = best
        if start is None or r < 0.68: continue  # only chunks that clearly are a script line
        pos = start + L
        win = list(range(max(0, start - 1), min(len(keys), start + L + 1)))
        changed = False
        for i, w in enumerate(cws):
            k = norm(w)
            if len(k) < 4 or k in keys[start:start + L]: continue
            j = max(win, key=lambda j: sim(k, keys[j]))
            if 0.55 <= sim(k, keys[j]) < 1 and len(keys[j]) >= 4:
                tail = re.search(r"[.,!?…]+$", w)
                cws[i] = words[j].strip(".,!?…\"'“”") + (tail.group(0) if tail else ""); changed = True
        if changed:
            new = " ".join(cws)
            c["text"] = case(new) if case else new; fixed += 1
    return captions, fixed
