"""Text normalisation for business names and addresses.

Design notes
* Country-agnostic: nothing here branches on {US, India}; France (unseen in training)
  goes through the same code. Language-specific abbreviation maps are plain expansions
  of common written forms (Rd/Road, R./Rue, Pvt/Private) -- domain knowledge in code,
  not an external data lookup.
* Non-Latin scripts (Devanagari, Telugu, Tamil, ...) are romanised with `anyascii`
  (ISC licence). An optional token dictionary learned from *training* pairs
  (see translit.py) is applied afterwards to map script tokens to their English form.
"""
import re
from typing import Dict, List, Tuple

from anyascii import anyascii

# ---------------------------------------------------------------- names
LEGAL = {
    "private": "pvt", "pvt": "pvt", "pte": "pvt", "praivet": "pvt", "prai": "pvt", "pra": "pvt",
    "limited": "ltd", "ltd": "ltd", "limitad": "ltd", "limited.": "ltd", "li": "ltd", "lim": "ltd",
    "incorporated": "inc", "inc": "inc",
    "corporation": "corp", "corp": "corp",
    "company": "co", "co": "co", "cos": "co",
    "llc": "llc", "llp": "llp", "elelpi": "llp", "lp": "lp", "plc": "plc", "pc": "pc",
    "pllc": "pllc", "public": "public", "opc": "opc",
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "sa": "sa", "eurl": "eurl", "sci": "sci",
    "snc": "snc", "scp": "scp", "selarl": "selarl", "scop": "scop", "gie": "gie",
    "gmbh": "gmbh", "ag": "ag",
}
HONORIFIC = {"mr", "mrs", "ms", "dr", "smt", "messrs", "mssrs", "m", "the"}
STOP = {"and", "of", "et", "de", "du", "des", "la", "le", "les", "l", "d", "a"}

_RE_DBA = re.compile(r"\b(?:trading as|doing business as|d/?b/?a|t/a|a\.?k\.?a\.?|formerly)\b", re.I)
_RE_URL = re.compile(r"(?:https?://)?(?:www\.)?([a-z0-9][a-z0-9-]*)\.(?:com|net|org|co\.in|in|fr|us|biz|info|io|co)\b")
_RE_MS = re.compile(r"\bm\s*/\s*s\b\.?")
_RE_NONWORD = re.compile(r"[^a-z0-9]+")


_RE_EDGE_PUNCT = re.compile(r"^[\W_]+|[\W_]+$")


def apply_tokmap(s: str, tokmap: Dict[str, str]) -> str:
    """Replace script tokens with their learned English form (punctuation-insensitive)."""
    out = []
    for t in s.replace(",", " , ").split():
        if t.isascii():
            out.append(t)
        else:
            out.append(tokmap.get(_RE_EDGE_PUNCT.sub("", t), t))
    return " ".join(out)


def fix_mojibake(s: str) -> str:
    """Repair UTF-8 text that was mis-decoded as Latin-1 ('Ã©' -> 'é', 'Â\\x80\\x99' -> '’')."""
    if "Ã" in s or "Â" in s:
        try:
            return s.encode("latin-1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return s.replace("Â", "")
    return s


def to_ascii_lower(s: str) -> str:
    """Romanise any script / strip accents, then lower-case."""
    if not s:
        return ""
    if not s.isascii():
        s = anyascii(fix_mojibake(s))
    return s.lower()


def _collapse_initials(toks: List[str]) -> List[str]:
    """Join runs of single letters: 'l l c' -> 'llc', 's a r l' -> 'sarl', 'p c' -> 'pc'."""
    out, run = [], []
    for t in toks:
        if len(t) == 1 and t.isalpha():
            run.append(t)
            continue
        if run:
            out.append("".join(run))
            run = []
        out.append(t)
    if run:
        out.append("".join(run))
    return out


def name_tokens(raw: str, tokmap: Dict[str, str] = None) -> Tuple[List[str], List[str]]:
    """Return (all canonical tokens, core tokens) for a business name.

    core = tokens minus legal forms, honorifics and stop words -- the part that
    identifies the business. Falls back to all tokens if the core would be empty.
    """
    s = raw
    if tokmap and not s.isascii():
        s = apply_tokmap(s, tokmap)  # learned script->English tokens before romanising
    s = to_ascii_lower(s)
    s = s.replace("&", " and ").replace("+", " plus ")
    s = _RE_MS.sub(" ", s)
    s = _RE_URL.sub(r" \1 ", s)
    s = s.replace("|", " ")
    toks = _collapse_initials([t for t in _RE_NONWORD.split(s) if t])
    toks = [LEGAL.get(t, t) for t in toks]
    legal_vals = set(LEGAL.values())
    core = [t for t in toks if t not in legal_vals and t not in HONORIFIC and t not in STOP]
    if not core:
        core = [t for t in toks if t not in HONORIFIC] or toks
    return toks, core


def split_dba(raw: str) -> List[str]:
    """'Jaxevo trading as Mansfield Devices' -> ['Jaxevo', 'Mansfield Devices']."""
    parts = [p.strip() for p in _RE_DBA.split(raw) if p and p.strip()]
    return parts or [raw]


# ---------------------------------------------------------------- addresses
ADDR_ABBR = {
    # English
    "street": "st", "str": "st", "road": "rd", "avenue": "ave", "av": "ave", "avn": "ave",
    "drive": "dr", "boulevard": "blvd", "lane": "ln", "court": "ct", "place": "pl",
    "circle": "cir", "highway": "hwy", "parkway": "pkwy", "terrace": "ter", "trail": "trl",
    "square": "sq", "suite": "ste", "apartment": "apt", "building": "bldg", "floor": "fl",
    "north": "n", "south": "s", "east": "e", "west": "w", "mount": "mt", "saint": "st",
    "number": "no", "nos": "no", "nagar": "ngr", "colony": "col", "sector": "sec",
    "opposite": "opp", "near": "nr", "behind": "bhd", "district": "dist", "dt": "dist",
    "house": "h", "hno": "h", "hn": "h", "plot": "plot", "cross": "crs", "main": "mn",
    "first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5",
    "1st": "1", "2nd": "2", "3rd": "3", "4th": "4", "5th": "5", "6th": "6", "7th": "7",
    "8th": "8", "9th": "9", "2st": "2", "2rd": "2",
    # French
    "rue": "rue", "r": "rue", "boulevard.": "blvd", "bd": "blvd", "bld": "blvd",
    "avenue.": "ave", "chemin": "ch", "chem": "ch", "allee": "all", "impasse": "imp",
    "route": "rte", "quai": "qu", "cours": "crs", "residence": "res", "batiment": "bat",
    "bis": "bis", "ter": "ter",
}
# Region names -> short codes, so 'Texas' == 'TX' and 'Tamil Nadu' == 'TN'.
# Multi-word names are joined before tokenising (see _RE_REGION).
REGIONS = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia",
    "kansas": "ks", "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn", "mississippi": "ms",
    "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv",
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny",
    "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok",
    "oregon": "or", "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc",
    "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt",
    "virginia": "va", "washington": "wa", "west virginia": "wv", "wisconsin": "wi",
    "wyoming": "wy", "district of columbia": "dc",
    # India (codes as used in the data, e.g. MH, TN, GJ, DL, TG, HR, KA)
    "andhra pradesh": "ap", "andhra": "ap", "arunachal pradesh": "ar_in", "assam": "as",
    "bihar": "br", "chhattisgarh": "cg", "goa": "ga_in", "gujarat": "gj", "haryana": "hr",
    "himachal pradesh": "hp", "jharkhand": "jh", "karnataka": "ka", "kerala": "kl",
    "madhya pradesh": "mp", "maharashtra": "mh", "manipur": "mn_in", "meghalaya": "ml",
    "mizoram": "mz", "nagaland": "nl", "odisha": "od", "orissa": "od", "punjab": "pb",
    "rajasthan": "rj", "sikkim": "sk", "tamil nadu": "tn_in", "tamilnadu": "tn_in",
    "tamil": "tn_in", "telangana": "tg", "tripura": "tr", "uttar pradesh": "up",
    "uttarakhand": "uk", "west bengal": "wb", "delhi": "dl", "chandigarh": "ch_in",
    "jammu and kashmir": "jk", "puducherry": "py", "pondicherry": "py",
}
_RE_REGION = re.compile(r"\b(" + "|".join(sorted((re.escape(k) for k in REGIONS), key=len, reverse=True)) + r")\b")
# India state codes that collide with US codes are disambiguated only when the long form
# is used; short codes stay as written ('tn' in an Indian address still means Tamil Nadu).
_CODE_ALIAS = {"tn_in": "tn", "ar_in": "ar", "ga_in": "ga", "mn_in": "mn", "ch_in": "ch"}

ADDR_DROP = {"null", "none", "na", "n/a", "the", "of", "and", "de", "du", "des", "la", "le",
             "les", "city", "town"}
_RE_POBOX = re.compile(r"\b(?:p\.?\s*o\.?\s*box|pmb|post box)\s*#?\s*(\d+)")


def address_tokens(raw: str, tokmap: Dict[str, str] = None) -> Tuple[List[str], List[str], str]:
    """Return (canonical tokens, digit tokens, po-box number) for an address."""
    s = raw
    if tokmap and not s.isascii():
        s = apply_tokmap(s, tokmap)
    s = to_ascii_lower(s)
    pobox = ""
    m = _RE_POBOX.search(s)
    if m:
        pobox = m.group(1)
        s = _RE_POBOX.sub(" ", s)
    s = _RE_REGION.sub(lambda m: " " + _CODE_ALIAS.get(REGIONS[m.group(1)], REGIONS[m.group(1)]) + " ", s)
    toks = [t for t in _RE_NONWORD.split(s) if t]
    toks = [ADDR_ABBR.get(t, t) for t in toks if t not in ADDR_DROP]
    digits = [t for t in toks if any(c.isdigit() for c in t)]
    return toks, digits, pobox


def normalize_record(name: str, address: str, tokmap: Dict[str, str] = None) -> dict:
    """All normalised views of one record, used by blocking and features."""
    parts = split_dba(name)
    toks, core = name_tokens(name, tokmap)
    alt_core = []
    if len(parts) > 1:
        for p in parts:
            alt_core.append(" ".join(name_tokens(p, tokmap)[1]))
    atoks, digits, pobox = address_tokens(address, tokmap)
    return {
        "name_norm": " ".join(toks),
        "name_core": " ".join(core),
        "name_alt": "|".join(alt_core),
        "addr_norm": " ".join(atoks),
        "addr_digits": " ".join(digits),
        "pobox": pobox,
    }
