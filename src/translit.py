"""Learn a script-token -> English-token dictionary from *training* match pairs.

Source-2/3 India records often spell the business name in Devanagari, Telugu, Tamil, ...
while Source 1 is always Latin. Generic romanisation (anyascii) gets close
('प्राइवेट' -> 'praivet') but not close enough for token matching. Here we align tokens
across known matching pairs (training fold only) and keep a mapping when
  * the pair co-occurs often (Dice coefficient over pairs), and
  * the romanised script token is phonetically similar to the English token,
so the dictionary is learned purely from the provided training data.
"""
import collections
import json
import re
from pathlib import Path
from typing import Dict, Iterable, Tuple

from anyascii import anyascii
from rapidfuzz.distance import JaroWinkler

_PUNCT = re.compile(r"^[\W_]+|[\W_]+$")
_LAT = re.compile(r"[a-z0-9]+")


def skeleton(s: str) -> str:
    """Rough phonetic key: romanise, drop vowels/h after first char, collapse repeats."""
    s = anyascii(s).lower()
    s = re.sub(r"[^a-z]", "", s)
    if not s:
        return ""
    s = s.replace("ph", "f").replace("sh", "s").replace("w", "v").replace("z", "j")
    head, tail = s[0], re.sub(r"[aeiouyh]", "", s[1:])
    out = head + tail
    return re.sub(r"(.)\1+", r"\1", out)


def script_tokens(text: str):
    for t in text.split():
        t = _PUNCT.sub("", t)
        if t and not t.isascii():
            yield t


def learn(pairs: Iterable[Tuple[str, str]], min_count=3, min_dice=0.25, min_sim=0.6) -> Dict[str, str]:
    """pairs = (latin_text_from_S1, script_text_from_S2_or_S3)."""
    c_t, c_w, c_tw = collections.Counter(), collections.Counter(), collections.Counter()
    for lat, scr in pairs:
        ts = set(script_tokens(scr))
        if not ts:
            continue
        ws = set(_LAT.findall(lat.lower()))
        c_t.update(ts)
        c_w.update(ws)
        for t in ts:
            for w in ws:
                c_tw[(t, w)] += 1
    best = {}
    for (t, w), n in c_tw.items():
        if c_t[t] < min_count:
            continue
        dice = 2 * n / (c_t[t] + c_w[w])
        if dice < min_dice:
            continue
        sim = JaroWinkler.similarity(skeleton(t), skeleton(w))
        if sim < min_sim:
            continue
        score = dice * sim
        if score > best.get(t, ("", 0.0))[1]:
            best[t] = (w, score)
    return {t: w for t, (w, _) in best.items()}


def save(mapping: Dict[str, str], path: Path) -> None:
    path.write_text(json.dumps(mapping, ensure_ascii=False, indent=0), encoding="utf-8")


def load(path: Path) -> Dict[str, str]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
