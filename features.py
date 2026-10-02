"""Pairwise features for (S1, S2/S3) candidate pairs."""
import math
import sys
import time
from collections import Counter
from multiprocessing import Pool

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

from config import WORK

STR_COLS = ["name_full", "core", "compact", "alias", "addr", "state", "nums", "name_lite", "addr_lite", "script"]
LEGAL_KEEP = {"inc", "corp", "co", "ltd", "pvt", "llc", "llp", "lp", "pc", "plc", "pllc", "sarl", "sas", "sasu",
              "sa", "sci", "eurl", "ei", "snc"}
STRING_FEATS = [
    "n_ratio", "n_tset", "n_tsort", "n_partial", "n_jw", "c_ratio", "c_partial", "full_tset",
    "n_jacc", "n_idf_jacc", "n_idf_cover1", "n_idf_cover2", "n_len1", "n_len2", "first_tok_eq",
    "alias_has", "alias_tset", "legal_eq", "legal_both",
    "a_tset", "a_jacc", "a_idf_jacc", "a_len1", "a_len2", "a_empty",
    "num_first_eq", "num_overlap", "num_jacc", "num_first_absdiff", "nums_both",
    "state_eq", "state_other_empty",
    # plan v2 F1: raw-lite vs normalized (guards against over-normalization); F5: other name was native script
    "nl_ratio", "nl_tset", "al_ratio", "al_tset", "script_o",
]
FEATURES = (["sim_name_blk", "sim_addr_blk"] + STRING_FEATS
            + ["is_s3", "rk_name_o", "rk_addr_o", "gap_name_o", "gap_addr_o", "n_cand_o",
               "rk_name_s", "rk_addr_s", "gap_name_s", "n_cand_s"])

_IDF = None


def _init(idf):
    global _IDF
    _IDF = idf


def build_idf(split):
    cnt = Counter()
    n = 0
    for s in (1, 2, 3):
        df = pl.read_parquet(WORK / f"{split}_s{s}_norm.parquet", columns=["core", "addr"])
        for col in ("core", "addr"):
            for toks in df[col].str.split(" ").to_list():
                cnt.update(set(toks))
        n += len(df)
    return {t: math.log(n / c) for t, c in cnt.items() if t}


def _idf_stats(a, b):
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0, 0.0, 0.0
    w = lambda s: sum(_IDF.get(t, 12.0) for t in s)
    inter = w(sa & sb)
    return inter / w(sa | sb), inter / w(sa), inter / w(sb)


def _jacc(sa, sb):
    return len(sa & sb) / len(sa | sb) if sa and sb else 0.0


def _pair_feats(r):
    (full1, core1, comp1, _al1, addr1, st1, nums1, nl1, al1, _sc1,
     full2, core2, comp2, al2, addr2, st2, nums2, nl2, al2_lite, sc2) = r
    t1, t2 = core1.split(), core2.split()
    s1, s2 = set(t1), set(t2)
    n_ratio = fuzz.ratio(core1, core2)
    n_tset = fuzz.token_set_ratio(core1, core2)
    alias_tset = fuzz.token_set_ratio(core1, al2) if al2 else 0.0
    idf_j, cov1, cov2 = _idf_stats(t1, t2)
    l1 = {t for t in full1.split() if t in LEGAL_KEEP}
    l2 = {t for t in full2.split() if t in LEGAL_KEEP}
    a1, a2 = addr1.split(), addr2.split()
    aj, _, _ = _idf_stats(a1, a2)
    n1, n2 = nums1.split(), nums2.split()
    sn1, sn2 = set(n1), set(n2)
    if n1 and n2:
        first_eq = float(n1[0] == n2[0])
        try:
            diff = math.log1p(abs(int(n1[0][:9]) - int(n2[0][:9])))
        except ValueError:
            diff = -1.0
    else:
        first_eq, diff = -1.0, -1.0
    st_a, st_b = set(st1.split("|")) - {""}, set(st2.split("|")) - {""}
    return (
        n_ratio, n_tset, fuzz.token_sort_ratio(core1, core2), fuzz.partial_ratio(core1, core2),
        JaroWinkler.normalized_similarity(comp1, comp2), fuzz.ratio(comp1, comp2), fuzz.partial_ratio(comp1, comp2),
        fuzz.token_set_ratio(full1, full2),
        _jacc(s1, s2), idf_j, cov1, cov2, len(t1), len(t2), float(bool(t1) and bool(t2) and t1[0] == t2[0]),
        float(bool(al2)), alias_tset, float(l1 == l2), float(bool(l1 & l2)),
        fuzz.token_set_ratio(addr1, addr2), _jacc(set(a1), set(a2)), aj, len(a1), len(a2), float(not a2),
        first_eq, float(len(sn1 & sn2)), _jacc(sn1, sn2), diff, float(bool(n1) and bool(n2)),
        float(bool(st_a & st_b)), float(not st_b),
        fuzz.ratio(nl1, nl2), fuzz.token_set_ratio(nl1, nl2),
        fuzz.ratio(al1, al2_lite) if al1 and al2_lite else -1.0,
        fuzz.token_set_ratio(al1, al2_lite) if al1 and al2_lite else -1.0,
        float(sc2 not in ("", "latin")),
    )


def _work(rows):
    # float32 array rather than a list of float tuples: ~8x smaller to pickle back to the parent. The tuple lists
    # failed on Windows pipes (OSError 22 in MaybeEncodingError) once the feature count grew.
    return np.asarray([_pair_feats(r) for r in rows], dtype=np.float32).reshape(-1, len(STRING_FEATS))


def context_features(cand):
    """Rank/gap of each pair among the candidates of the same S2/S3 record and of the same S1 record."""
    return cand.with_columns(
        pl.col("sim_name_blk").rank("min", descending=True).over("other").alias("rk_name_o"),
        pl.col("sim_addr_blk").rank("min", descending=True).over("other").alias("rk_addr_o"),
        (pl.col("sim_name_blk").max().over("other") - pl.col("sim_name_blk")).alias("gap_name_o"),
        (pl.col("sim_addr_blk").max().over("other") - pl.col("sim_addr_blk")).alias("gap_addr_o"),
        pl.len().over("other").alias("n_cand_o"),
        pl.col("sim_name_blk").rank("min", descending=True).over("s1").alias("rk_name_s"),
        pl.col("sim_addr_blk").rank("min", descending=True).over("s1").alias("rk_addr_s"),
        (pl.col("sim_name_blk").max().over("s1") - pl.col("sim_name_blk")).alias("gap_name_s"),
        pl.len().over("s1").alias("n_cand_s"),
        pl.col("other").str.starts_with("S3").cast(pl.Float32).alias("is_s3"),
    )


def build(split, keep=None, out_name=None, chunk=1_000_000, cand_name=None, resume=False):
    """keep: optional callable(cand) -> filtered cand, applied after global context features.
    resume: skip parts already on disk; only valid when keep() returns a deterministically ordered frame."""
    cand = context_features(pl.read_parquet(WORK / (cand_name or f"{split}_cand.parquet")))
    if keep is not None:
        cand = keep(cand)
    s1 = pl.read_parquet(WORK / f"{split}_s1_norm.parquet", columns=["entity_id"] + STR_COLS)
    other = pl.concat([pl.read_parquet(WORK / f"{split}_s{i}_norm.parquet", columns=["entity_id"] + STR_COLS)
                       for i in (2, 3)])
    st = pl.read_parquet(WORK / f"{split}_other_state.parquet")
    other = other.drop("state").join(st, on="entity_id")
    idf = build_idf(split)
    out_dir = WORK / (out_name or f"{split}_feats")
    out_dir.mkdir(exist_ok=True)
    with Pool(11, initializer=_init, initargs=(idf,)) as pool:
        for ci, i in enumerate(range(0, len(cand), chunk)):
            dest = out_dir / f"part{ci:04d}.parquet"
            if resume and dest.exists():
                continue
            t = time.time()
            part = cand.slice(i, chunk)
            j = (part.select("s1", "other")
                     .join(s1, left_on="s1", right_on="entity_id", how="left")
                     .join(other, left_on="other", right_on="entity_id", how="left", suffix="_o"))
            rows = j.select(STR_COLS + [c + "_o" for c in STR_COLS]).fill_null("").rows()
            sub = [rows[k:k + 20000] for k in range(0, len(rows), 20000)]
            feats = np.concatenate(list(pool.imap(_work, sub)))
            fdf = pl.DataFrame(feats, schema=STRING_FEATS)
            outp = pl.concat([part.select(["s1", "other"] + [c for c in FEATURES if c not in STRING_FEATS]),
                              fdf], how="horizontal")
            tmp = dest.with_suffix(".tmp")  # write-then-rename: a crash never leaves a truncated part behind
            outp.with_columns(pl.col(c).cast(pl.Float32) for c in FEATURES).write_parquet(tmp)
            tmp.replace(dest)
            print(split, "chunk", ci, len(part), f"{time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    split = sys.argv[1] if len(sys.argv) > 1 else "train"
    tag = sys.argv[2] if len(sys.argv) > 2 else ""
    sfx = f"_{tag}" if tag else ""
    if split == "train":
        # S1 ids hashed into buckets: 0-2 -> model training (30%), 9 -> validation (10%)
        bucket = pl.col("s1").hash(7) % 10
        cname = f"train{sfx}_cand.parquet"
        if "--rest" in sys.argv:
            # every pair not covered by the two sets below, so that all train pairs have features
            def rest(c):
                val_o = c.filter(bucket == 9).select("other").unique()
                # sorted so chunk boundaries are identical across runs, which makes resume safe
                return c.filter(bucket >= 3).join(val_o, on="other", how="anti").sort("s1", "other")
            build("train", rest, f"rest{sfx}_feats", cand_name=cname, resume=True)
            sys.exit()
        # a filter keeps the candidate file's row order, so resuming the train parts is safe
        build("train", lambda c: c.filter(bucket < 3), f"train{sfx}_feats", cand_name=cname, resume=True)
        # validation keeps every competing candidate of the S2/S3 records touching validation S1s,
        # so the one-owner assignment is evaluated exactly as at inference time
        build("train", lambda c: c.join(c.filter(bucket == 9).select("other").unique(), on="other", how="semi"),
              f"val{sfx}_feats", cand_name=cname)
    else:
        build(split, out_name=f"{split}{sfx}_feats", cand_name=f"{split}{sfx}_cand.parquet")
