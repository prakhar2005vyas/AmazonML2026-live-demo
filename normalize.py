"""Record normalization: names (transliteration, aliases, legal suffixes) and addresses (state, numbers, tokens)."""
import json
import os
import re
import sys
import time
import unicodedata
from multiprocessing import Pool

import polars as pl
from indic_transliteration import sanscript
from unidecode import unidecode

from config import WORK

NON_LATIN = re.compile(r"[ऀ-෿]")
TOKSEP = re.compile(r"[\s,.;:()\[\]{}\-/&'\"#@!?+*|<>_=~`$%^]+")
SCRIPTS = [  # (first codepoint, last codepoint, sanscript scheme)
    (0x0900, 0x097F, sanscript.DEVANAGARI), (0x0980, 0x09FF, sanscript.BENGALI),
    (0x0A00, 0x0A7F, sanscript.GURMUKHI), (0x0A80, 0x0AFF, sanscript.GUJARATI),
    (0x0B00, 0x0B7F, sanscript.ORIYA), (0x0B80, 0x0BFF, sanscript.TAMIL),
    (0x0C00, 0x0C7F, sanscript.TELUGU), (0x0C80, 0x0CFF, sanscript.KANNADA),
    (0x0D00, 0x0D7F, sanscript.MALAYALAM),
]

ALIAS = re.compile(r"\b(?:d\s*/?\s*b\s*/?\s*a|formerly(?:\s+known\s+as)?|f\s*/\s*k\s*/\s*a|fka|a\s*/\s*k\s*/\s*a|aka|trading\s+as|t\s*/\s*a)\b\s*:?")
ID_TAG = re.compile(r"\(?\s*\bid\s*:?\s*\d+\s*\)?")
MS_PREFIX = re.compile(r"\bm\s*/\s*s\b\.?")
DOTTED = re.compile(r"(?<=\b[a-z])\.(?=[a-z]\b)")
WEBSITE = re.compile(r"\b(?:www\.)?([a-z0-9][a-z0-9-]*)\.(?:com|net|org|in|co|biz|info|fr|us|io)\b")
NONALNUM = re.compile(r"[^a-z0-9]+")
LEET = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "8": "b"})

NAME_CANON = {
    "incorporated": "inc", "corporation": "corp", "limited": "ltd", "private": "pvt", "company": "co",
    "pvtltd": "pvt ltd", "llc": "llc", "l": "l", "associes": "associes", "cie": "co", "compagnie": "co",
    # plan v2 N5: French forms (France is test-only, so these have to be engineered in)
    "etablissements": "ets", "etablissement": "ets", "etab": "ets", "freres": "freres", "frere": "freres",
}
LEGAL = {"inc", "corp", "co", "ltd", "pvt", "llc", "llp", "lp", "pc", "plc", "pllc", "sarl", "sas", "sasu",
         "sa", "sci", "eurl", "ei", "snc", "gmbh", "and", "the", "of", "india", "france", "ms", "et", "cie",
         # plan v2 N5: French filler words, handled like "the" / "of" / "and"
         "de", "du", "des", "la", "le", "les"}

ADDR_CANON = {
    "street": "st", "saint": "st", "sainte": "ste", "road": "rd", "avenue": "ave", "av": "ave",
    "boulevard": "blvd", "bd": "blvd", "bld": "blvd", "boul": "blvd", "drive": "dr", "lane": "ln",
    "court": "ct", "circle": "cir", "place": "pl", "highway": "hwy", "parkway": "pkwy", "terrace": "ter",
    "trail": "trl", "square": "sq", "north": "n", "south": "s", "east": "e", "west": "w",
    "apartment": "apt", "appartement": "apt", "suite": "ste", "floor": "fl", "flr": "fl", "ground": "grd",
    "first": "1st", "second": "2nd", "third": "3rd", "building": "bldg", "near": "nr", "opposite": "opp",
    "sector": "sec", "station": "stn", "railway": "rly", "complex": "cmplx", "r": "rue", "route": "rte",
    "chemin": "chem", "all": "allee", "impasse": "imp", "mount": "mt", "fort": "ft", "bengaluru": "bangalore",
    "gurugram": "gurgaon", "calcutta": "kolkata", "bombay": "mumbai", "trivandrum": "thiruvananthapuram",
}
ADDR_STOP = {"no", "nos", "number", "door", "hno", "dno", "of", "the", "de", "du", "des", "la", "le", "les",
             "and", "city", "null", "na", "n", "a", "d", "l", "unit", "at", "post", "c", "o", "h", "po"}
ADDR_STOP -= {"n", "a", "d"}  # direction / block letters carry signal
NULLISH = {"null", "<null>", "n/a", "na", "none", ""}

US_STATES = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas", "ca": "california", "co": "colorado",
    "ct": "connecticut", "de": "delaware", "dc": "district of columbia", "fl": "florida", "ga": "georgia",
    "hi": "hawaii", "id": "idaho", "il": "illinois", "in": "indiana", "ia": "iowa", "ks": "kansas",
    "ky": "kentucky", "la": "louisiana", "me": "maine", "md": "maryland", "ma": "massachusetts",
    "mi": "michigan", "mn": "minnesota", "ms": "mississippi", "mo": "missouri", "mt": "montana",
    "ne": "nebraska", "nv": "nevada", "nh": "new hampshire", "nj": "new jersey", "nm": "new mexico",
    "ny": "new york", "nc": "north carolina", "nd": "north dakota", "oh": "ohio", "ok": "oklahoma",
    "or": "oregon", "pa": "pennsylvania", "ri": "rhode island", "sc": "south carolina", "sd": "south dakota",
    "tn": "tennessee", "tx": "texas", "ut": "utah", "vt": "vermont", "va": "virginia", "wa": "washington",
    "wv": "west virginia", "wi": "wisconsin", "wy": "wyoming", "pr": "puerto rico",
}
IN_STATES = {
    "mh": "maharashtra", "dl": "delhi", "up": "uttar pradesh", "ka": "karnataka", "tn": "tamil nadu",
    "wb": "west bengal", "gj": "gujarat", "tg": "telangana", "ts": "telangana", "hr": "haryana",
    "rj": "rajasthan", "kl": "kerala", "br": "bihar", "mp": "madhya pradesh", "ap": "andhra pradesh",
    "pb": "punjab", "od": "odisha", "or": "odisha", "ga": "goa", "as": "assam", "jh": "jharkhand",
    "ct": "chhattisgarh", "cg": "chhattisgarh", "uk": "uttarakhand", "ut": "uttarakhand",
    "hp": "himachal pradesh", "jk": "jammu and kashmir", "ch": "chandigarh", "py": "puducherry",
    "orissa": "odisha", "keralam": "kerala", "pondicherry": "puducherry", "uttaranchal": "uttarakhand",
}
FR_REGIONS = {
    "nord": "hauts de france", "pas de calais": "hauts de france", "somme": "hauts de france",
    "aisne": "hauts de france", "oise": "hauts de france", "gironde": "nouvelle aquitaine",
    "landes": "nouvelle aquitaine", "dordogne": "nouvelle aquitaine", "pyrenees atlantiques": "nouvelle aquitaine",
    "loire atlantique": "pays de la loire", "vendee": "pays de la loire", "maine et loire": "pays de la loire",
    "sarthe": "pays de la loire", "mayenne": "pays de la loire",
}


def _state_table():
    table = {}
    for code, name in US_STATES.items():
        table[("us", code)] = name
        table[("us", name)] = name
    for code, name in IN_STATES.items():
        table[("india", code)] = name
        table[("india", name)] = name
    for name in set(IN_STATES.values()):
        table[("india", name)] = name
    for dep, reg in FR_REGIONS.items():
        table[("france", dep)] = reg
        table[("france", reg)] = reg
    return table


STATE_TABLE = _state_table()
_TR = None


def translit_tables():
    global _TR
    if _TR is None:
        with open(WORK / "translit.json", encoding="utf-8") as f:
            _TR = json.load(f)
    return _TR


def _script_of(tok):
    for ch in tok:
        cp = ord(ch)
        for lo, hi, scheme in SCRIPTS:
            if lo <= cp <= hi:
                return scheme
    return None


# plan v2 N2: fold the scholarly ITRANS output toward colloquial English spelling ("garamentsa" -> "garments")
FOLD = True
CANDRA_O = str.maketrans({"ऑ": "ओ", "ॉ": "ो"})  # ऑ / ॉ (English loan "o"), unknown to sanscript
SCHWA_MEDIAL = {sanscript.DEVANAGARI, sanscript.GURMUKHI, sanscript.GUJARATI}  # Hindi-type schwa deletion
SCHWA_FINAL = SCHWA_MEDIAL | {sanscript.BENGALI, sanscript.ORIYA}  # Dravidian final vowels are real: left alone
V = "aeiou"
MEDIAL = re.compile(rf"(?<=[{V}][^{V}])a(?=[^{V}h][{V}])")
FINAL = re.compile(rf"(?<=[^{V}])a$")


def fold_itrans(s, scheme):
    s = re.sub(r"M(?=[pbm]|$)", "m", s).replace("M", "n")  # anusvara: m before labials and at word end ("om")
    s = s.replace(".N", "").replace(".h", "").replace("~N", "n").replace("~n", "n").replace("RRi", "ri").replace(".D", "d")
    s = re.sub(r"[^a-z]", "", s.lower())
    if scheme in SCHWA_FINAL and sum(c in V for c in s) >= 2:
        s = FINAL.sub("", s)
    if scheme in SCHWA_MEDIAL:
        # right to left, as in Hindi schwa deletion: VCaCV -> VCCV
        for m in reversed(list(MEDIAL.finditer(s))):
            s = s[:m.start()] + s[m.end():]
    return s


def translit_token(tok, table):
    key = TOKSEP.sub("", tok.lower())
    if key in table:
        return table[key]
    scheme = _script_of(tok)
    if scheme is None:
        return tok
    if FOLD:
        return fold_itrans(sanscript.transliterate(key.translate(CANDRA_O), scheme, sanscript.ITRANS), scheme)
    return sanscript.transliterate(key, scheme, sanscript.ITRANS).lower()


def to_ascii(s, table):
    s = unicodedata.normalize("NFKC", s)
    if NON_LATIN.search(s):
        s = " ".join(translit_token(t, table) if NON_LATIN.search(t) else t for t in s.split())
    return unidecode(s).lower()


def _deleet(tok):
    if tok.isdigit() or tok.isalpha():
        return tok
    letters = sum(c.isalpha() for c in tok)
    return tok.translate(LEET) if letters >= 2 * (len(tok) - letters) else tok


def name_tokens(s):
    s = ID_TAG.sub(" ", s)
    s = MS_PREFIX.sub(" ", s)
    s = s.replace("&", " and ").replace("+", " and ")
    s = DOTTED.sub("", s)
    s = WEBSITE.sub(r"\1", s)
    out = []
    for t in NONALNUM.sub(" ", s).split():
        t = NAME_CANON.get(_deleet(t), _deleet(t))
        out.extend(t.split())
    return out


# --- plan v2 stage 3: France pack. France only occurs in test, so these rules change no training row and
# need no retraining. ER_FR_PACK=0 turns the pack off.
FR_PACK = os.environ.get("ER_FR_PACK", "1") == "1"
FR_NAME_NOISE = {"fils", "associes", "associe", "associees", "groupe", "freres", "frere"}  # "& Fils", "Et Fils", "& Associés"
FR_ADDR_CANON = {"ch": "chem", "che": "chem", "crs": "cours", "qu": "quai", "fbg": "faubourg", "faub": "faubourg",
                 "res": "residence", "resid": "residence", "psg": "passage", "pass": "passage",
                 "bis": "b", "ter": "t"}  # "3 BIS" and "3B" both become number 3 + token b
FR_HOUSE_SFX = re.compile(r"(\d+)(bis|ter|b|t)")
DEGREE = str.maketrans({"°": "o", "º": "o"})  # "N°230": unidecode would give "Ndeg230"


def norm_name(raw, table, country=""):
    s = to_ascii(raw, table)
    legal = LEGAL | FR_NAME_NOISE if FR_PACK and country == "france" else LEGAL
    parts = [p for p in ALIAS.split(s) if p.strip()]
    full = name_tokens(s)
    core = [t for t in full if t not in legal]
    alias = []
    if len(parts) > 1:
        alias = [t for t in name_tokens(parts[-1]) if t not in legal]
        primary = [t for t in name_tokens(parts[0]) if t not in legal]
        core = primary
    return " ".join(full), " ".join(core), "".join(core), " ".join(alias)


ORDINAL = re.compile(r"\d+(?:st|nd|rd|th)")
NUM_WORD = re.compile(r"(\d+(?:st|nd|rd|th)?)(?=[a-z]{3,})")  # "4gitanjaliplaza" -> "4 gitanjaliplaza", "2ndfloor" -> "2nd floor"
WORD_NUM = re.compile(r"(?<=[a-z]{3})(?=\d)")                 # "flat502" -> "flat 502", "sector120" -> "sector 120"


def unglue(t):
    """plan v2 N1: split digits glued to words. Pure words, pure numbers and ordinals stay as they are."""
    if t.isalpha() or t.isdigit() or ORDINAL.fullmatch(t):
        return [t]
    return WORD_NUM.sub(" ", NUM_WORD.sub(r"\1 ", t)).split()


def script_of_text(s):
    """Script of the first native-script character in s, or 'latin' (plan v2 N2/F5)."""
    for ch in s:
        cp = ord(ch)
        for lo, hi, scheme in SCRIPTS:
            if lo <= cp <= hi:
                return scheme
    return "latin"


def lite(raw, table):
    """plan v2 N3: light cleaning only (ascii, lowercase, punctuation to space). Nothing is dropped or reordered."""
    return " ".join(NONALNUM.sub(" ", to_ascii(raw, table)).split())


FR_NO_NUM = re.compile(r"no(\d+)")  # "N°230" -> "no230" -> "230"


def fr_tokens(t):
    m = FR_NO_NUM.fullmatch(t)
    if m:
        return [m.group(1)]
    m = FR_HOUSE_SFX.fullmatch(t)
    return [m.group(1), FR_ADDR_CANON.get(m.group(2), m.group(2))] if m else [FR_ADDR_CANON.get(t, t)]


def norm_addr(raw, country, table):
    fr = FR_PACK and country == "france"
    s = unicodedata.normalize("NFKC", raw.translate(DEGREE) if fr else raw)
    comps = []
    for c in s.split(","):
        c = c.strip()
        key = " ".join(t for t in TOKSEP.split(c.lower()) if t)
        if key in table:
            c = table[key]
        comps.append(unidecode(c).lower().strip())
    comps = [c for c in comps if c not in NULLISH]
    states = []
    rest = []
    for c in comps:
        key = NONALNUM.sub(" ", c).strip()
        st = STATE_TABLE.get((country, key))
        if st:
            if st not in states:
                states.append(st)
        else:
            rest.append(key)
    # ambiguous city/state names (e.g. "Delaware, Ohio") keep every candidate state, "|"-separated
    state = "|".join(states)
    toks, nums = [], []
    for c in rest:
        words = (u for t0 in c.split() for u in unglue(t0))
        if fr:
            words = (v for u in words for v in fr_tokens(u))
        for t in words:
            for d in re.findall(r"\d+", t):
                d = d.lstrip("0") or "0"
                if d not in nums:
                    nums.append(d)
            t = ADDR_CANON.get(t, t)
            if t not in ADDR_STOP and not t.isdigit():
                toks.append(t)
    return " ".join(toks), state, " ".join(nums)


def _work(rows):
    tr = translit_tables()
    out = []
    for eid, name, addr, country in rows:
        c = country.strip().lower()
        full, core, compact, alias = norm_name(name, tr["name"], c)
        atoks, state, nums = norm_addr(addr, c, tr["addr"])
        out.append((eid, c, full, core, compact, alias, atoks, state, nums,
                    lite(name, tr["name"]), lite(addr, tr["name"]), script_of_text(name)))
    return out


COLS = ["entity_id", "country", "name_full", "core", "compact", "alias", "addr", "state", "nums",
        "name_lite", "addr_lite", "script"]


def normalize_frame(df, pool, chunk=20000):
    rows = list(df.select("entity_id", "business_name", "business_address", "country").iter_rows())
    chunks = [rows[i:i + chunk] for i in range(0, len(rows), chunk)]
    res = [r for part in pool.imap(_work, chunks) for r in part]
    return pl.DataFrame(res, schema=COLS, orient="row")


def main(splits):
    with Pool(11) as pool:
        for split in splits:
            for s in (1, 2, 3):
                t = time.time()
                df = pl.read_parquet(WORK / f"{split}_s{s}.parquet")
                out = normalize_frame(df, pool)
                out.write_parquet(WORK / f"{split}_s{s}_norm.parquet")
                print(split, s, len(out), f"{time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["train", "test"])
