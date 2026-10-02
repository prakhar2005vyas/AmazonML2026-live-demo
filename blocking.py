"""Candidate generation.

For every S2/S3 record retrieve its top-K S1 records under two TF-IDF views, inside (country, state) blocks:
  * name view: char 3-grams of the compact core name (typo / spacing / website robust)
  * address view: word tokens of core name + address + house numbers + state
S2/S3 records with no parsable state are matched country-wide on the name view only.
"""
import os
import sys
import time

import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

from config import WORK

K_NAME = 3
K_ADDR = 3
K_NOSTATE = 5
# v2 pipeline: records without a state are searched country-wide, where 5 name neighbours miss ~24% of their
# true S1s (val); 20 char-gram + 20 word neighbours recover most of them (75.6% -> 89.4% pair recall)
WIDE_TAGS = {"hard2", "v2", "hard3", "v3"}
K_NOSTATE_WIDE = 20
# plan v2 B1 (stage 4): records with no state are, in train, exactly the empty-address ones, and they cause 61% of
# missed pairs (blocking recall 0.75). tags hard3/v3 give them a wider country-wide name search.
K_NOSTATE_BY_TAG = {"hard3": int(os.environ.get("ER_K_NOSTATE", "50")), "v3": int(os.environ.get("ER_K_NOSTATE", "50"))}
MAX_DF = 0.02
CHUNK = 200_000
BLOCK_MERGE = {"telangana": "andhra pradesh"}  # records swap these two freely


def name_text():
    return pl.when(pl.col("compact") != "").then(pl.col("compact")).otherwise(pl.col("name_full")).alias("t")


def addr_text():
    nums = pl.col("nums").str.replace_all(r"(\S+)", "#$1")
    return pl.concat_str([pl.col("core"), pl.col("alias"), pl.col("addr"), nums], separator=" ").alias("t")


def topk(s1_text, q_text, k, analyzer):
    if analyzer == "char":
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), max_df=MAX_DF, sublinear_tf=True,
                              dtype=np.float32)
    else:
        vec = TfidfVectorizer(analyzer="word", token_pattern=r"\S+", max_df=MAX_DF, sublinear_tf=True,
                              dtype=np.float32)
    try:
        bt = vec.fit_transform(s1_text).T.tocsr()
    except ValueError:  # tiny block with empty vocabulary after max_df pruning
        vec.set_params(max_df=1.0)
        bt = vec.fit_transform(s1_text).T.tocsr()
    rows, cols, vals = [], [], []
    for i in range(0, len(q_text), CHUNK):
        q = vec.transform(q_text[i:i + CHUNK])
        m = sp_matmul_topn(q, bt, top_n=k, threshold=0.05, n_threads=11).tocoo()
        rows.append(m.row + i)
        cols.append(m.col)
        vals.append(m.data)
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)


def name_queries(other):
    return pl.concat([
        other.select("entity_id", name_text()),
        other.filter(pl.col("alias") != "").select("entity_id", pl.col("alias").str.replace_all(" ", "").alias("t")),
    ])


def search_name(s1, other, k):
    q = name_queries(other)
    r, c, v = topk(s1.select(name_text())["t"].to_list(), q["t"].to_list(), k, "char")
    return pl.DataFrame({"s1": s1["entity_id"].to_numpy()[c], "other": q["entity_id"].to_numpy()[r],
                         "sim_name_blk": v})


def search_name_word(s1, other, k):
    """Word view of the core name: robust to token reordering and to char-gram noise from repeated affixes."""
    q = pl.concat([other.select("entity_id", pl.col("core").alias("t")),
                   other.filter(pl.col("alias") != "").select("entity_id", pl.col("alias").alias("t"))])
    vec = TfidfVectorizer(analyzer="word", token_pattern=r"\S+", sublinear_tf=True, dtype=np.float32)
    try:
        bt = vec.fit_transform(s1["core"].to_list()).T.tocsr()
    except ValueError:
        return pl.DataFrame(schema={"s1": pl.String, "other": pl.String, "sim_name_blk": pl.Float32})
    rows, cols, vals = [], [], []
    texts = q["t"].to_list()
    for i in range(0, len(texts), CHUNK):
        m = sp_matmul_topn(vec.transform(texts[i:i + CHUNK]), bt, top_n=k, threshold=0.05, n_threads=11).tocoo()
        rows.append(m.row + i)
        cols.append(m.col)
        vals.append(m.data)
    r, c, v = np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)
    # scored as a name similarity so downstream features keep their meaning; char and word views merge by max
    return pl.DataFrame({"s1": s1["entity_id"].to_numpy()[c], "other": q["entity_id"].to_numpy()[r],
                         "sim_name_blk": v})


def search_addr(s1, other, k):
    r, c, v = topk(s1.select(addr_text())["t"].to_list(), other.select(addr_text())["t"].to_list(), k, "word")
    return pl.DataFrame({"s1": s1["entity_id"].to_numpy()[c], "other": other["entity_id"].to_numpy()[r],
                         "sim_addr_blk": v})


def comp_expr():
    return (pl.col("business_address").str.to_lowercase().str.split(",")
              .list.eval(pl.element().str.replace_all(r"[^\p{L}0-9]+", " ").str.strip_chars()))


def infer_missing_states(split, s1, other):
    """Learn address-component -> state from S1 (same split) and fill S2/S3 records that have no state."""
    raw1 = pl.read_parquet(WORK / f"{split}_s1.parquet").select("entity_id", comp_expr().alias("comp"))
    m = (raw1.join(s1.select("entity_id", "country", "state"), on="entity_id")
             .filter(~pl.col("state").str.contains(r"\|"))
             .explode("comp").filter(pl.col("comp") != "")
             .group_by("country", "comp", "state").len()
             .with_columns((pl.col("len") / pl.col("len").sum().over("country", "comp")).alias("share"),
                           pl.col("len").sum().over("country", "comp").alias("tot"))
             .filter((pl.col("share") >= 0.95) & (pl.col("tot") >= 5))
             .select("country", "comp", pl.col("state").alias("inferred")))
    missing = other.filter(pl.col("state") == "").select("entity_id", "country")
    raw = pl.concat([pl.read_parquet(WORK / f"{split}_s{i}.parquet") for i in (2, 3)]).select(
        "entity_id", comp_expr().alias("comp"))
    inferred = (missing.join(raw, on="entity_id").explode("comp")
                       .join(m, on=["country", "comp"])
                       .group_by("entity_id").agg(pl.col("inferred").mode().first()))
    print(split, "records without state", len(missing), "state inferred for", len(inferred), flush=True)
    return (other.join(inferred, on="entity_id", how="left")
                 .with_columns(pl.when(pl.col("state") == "").then(pl.col("inferred").fill_null(""))
                                 .otherwise(pl.col("state")).alias("state"))
                 .drop("inferred"))


def with_blocks(df):
    blk = pl.col("state").str.split("|").list.eval(pl.element().replace(BLOCK_MERGE)).list.unique()
    return df.with_columns(blk.alias("block")).explode("block")


def run(split, tag=None):
    """tag='hard': drop HARD_DROP of train S1 so their matches become distractors, mimicking test density."""
    s1_norm = pl.read_parquet(WORK / f"{split}_s1_norm.parquet")
    wide = tag in WIDE_TAGS
    if tag and tag.startswith("hard"):
        s1_norm = s1_norm.filter(hard_keep())
    other = pl.concat([pl.read_parquet(WORK / f"{split}_s{i}_norm.parquet") for i in (2, 3)])
    other = infer_missing_states(split, s1_norm, other)
    other.select("entity_id", "state").write_parquet(WORK / f"{split}_other_state.parquet")
    s1_all = with_blocks(s1_norm)
    other_all = with_blocks(other)
    del s1_norm, other
    parts = []
    for country in s1_all["country"].unique().sort().to_list():
        t = time.time()
        s1c = s1_all.filter(pl.col("country") == country)
        oc = other_all.filter(pl.col("country") == country)
        n_before = sum(len(p) for p in parts)
        for block in s1c["block"].unique().drop_nulls().sort().to_list():
            a = s1c.filter(pl.col("block") == block)
            b = oc.filter(pl.col("block") == block)
            if len(b) == 0:
                continue
            parts.append(search_name(a, b, K_NAME))
            parts.append(search_addr(a, b, K_ADDR))
        no_state = oc.filter(pl.col("block").is_null() | (pl.col("block") == ""))
        s1_unique = s1c.unique("entity_id", keep="first")
        if len(no_state):
            ns = no_state.unique("entity_id")
            k_wide = K_NOSTATE_BY_TAG.get(tag, K_NOSTATE_WIDE)
            parts.append(search_name(s1_unique, ns, k_wide if wide else K_NOSTATE))
            if wide:
                parts.append(search_name_word(s1_unique, ns, k_wide))
        print(split, country, "S1", s1_unique.height, "others", oc["entity_id"].n_unique(), "no-state", len(no_state),
              "raw pairs", sum(len(p) for p in parts) - n_before, f"{time.time() - t:.0f}s", flush=True)
    cand = (pl.concat(parts, how="diagonal").fill_null(0.0)
              .group_by("s1", "other").agg(pl.col("sim_name_blk").max(), pl.col("sim_addr_blk").max()))
    cand.write_parquet(WORK / f"{split}{'_' + tag if tag else ''}_cand.parquet")
    return cand


# Train has ~1.2 unmatched S2/S3 records per S1, test ~2.3; dropping 18% of S1 reproduces the test ratio.
HARD_DROP = 18


def hard_keep(col="entity_id"):
    return pl.col(col).hash(11) % 100 >= HARD_DROP


def report(cand):
    gt = pl.read_parquet(WORK / "train_pairs.parquet").join(cand.select("s1").unique(), on="s1", how="semi")
    hit = gt.join(cand, on=["s1", "other"], how="semi")
    n_s1 = cand["s1"].n_unique()
    print(f"pair recall {len(hit) / len(gt):.4f}  candidates {len(cand)}  per-S1 {len(cand) / n_s1:.1f}")


if __name__ == "__main__":
    split = sys.argv[1] if len(sys.argv) > 1 else "train"
    tag = sys.argv[2] if len(sys.argv) > 2 else None
    cand = run(split, tag)
    if split == "train":
        report(cand)
