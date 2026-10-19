"""Gene-action encodings from independent modalities (design A.5).

Every encoding is a frozen vector per human gene, keyed by Ensembl gene ID through HGNC, and
built without any MitoCheck data (so no label leaks from the screen being predicted):

  perturbseq  Replogle et al. 2022 K562 genome-wide CRISPRi pseudobulk (CC BY 4.0): each
              perturbed gene's normalised expression response, PCA to 128 dimensions.
  jump_crispr JUMP-CP CRISPR knockout Cell Painting profiles (CC0; harmony-corrected PCA
              profiles), averaged over wells per gene, standardised.
  jump_orf    JUMP-CP ORF overexpression profiles (CC0), same treatment.
  string      STRING v12 human network (CC BY 4.0), combined score >= 400; spectral
              embedding (top 128 eigenvectors of the normalised adjacency).
  esm2        ESM-2 650M (MIT) mean-pooled embedding of the reviewed UniProt canonical protein
              (first 1,022 residues), PCA to 128 dimensions.

Outputs `data/actions/<source>.npz` (genes, X) and `results/actions_coverage.json`
(how many MitoCheck stage-1 and split genes each source covers).
"""
from __future__ import annotations

import gzip
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "actions" / "raw"
OUT = REPO / "data" / "actions"
DIM = 128


def hgnc() -> pd.DataFrame:
    h = pd.read_csv(RAW / "hgnc_complete_set.txt", sep="\t", low_memory=False)
    return h[["symbol", "ensembl_gene_id", "entrez_id", "uniprot_ids", "prev_symbol",
              "alias_symbol"]]


def symbol_to_ensg(h: pd.DataFrame) -> dict[str, str]:
    m = {}
    for col in ("alias_symbol", "prev_symbol", "symbol"):  # later columns win
        for s, e in zip(h[col], h.ensembl_gene_id, strict=False):
            if isinstance(s, str) and isinstance(e, str):
                for x in s.split("|"):
                    m[x.upper()] = e
    return m


def pca(x: np.ndarray, k: int = DIM) -> np.ndarray:
    x = x - x.mean(0)
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    return (u[:, :k] * s[:k]).astype(np.float32)


def standardise(x: np.ndarray) -> np.ndarray:
    """Column z-score; columns with no finite values are dropped, remaining NaNs set to 0."""
    x = np.asarray(x, np.float64)
    x[~np.isfinite(x)] = np.nan
    x = x[:, ~np.all(np.isnan(x), 0)]
    z = (x - np.nanmean(x, 0)) / (np.nanstd(x, 0) + 1e-6)
    return np.nan_to_num(z, nan=0.0).astype(np.float32)


def perturbseq() -> tuple[list[str], np.ndarray]:
    import anndata as ad
    a = ad.read_h5ad(RAW / "replogle_k562_gwps_bulk.h5ad")
    ensg = [t.split("_")[-1] for t in a.obs_names]
    keep = [i for i, e in enumerate(ensg) if e.startswith("ENSG")]
    x = np.asarray(a.X)[keep]
    genes = [ensg[i] for i in keep]
    df = pd.DataFrame(x, index=genes).groupby(level=0).mean()
    return list(df.index), pca(standardise(df.to_numpy()))


def jump(kind: str, h: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    import pyarrow.parquet as pq
    meta = pd.read_csv(RAW / f"jump_{kind}.csv.gz")
    prof = pq.read_table(RAW / f"jump_{kind}_pca.parquet").to_pandas()
    feats = [c for c in prof.columns if not c.startswith("Metadata_")]
    prof = prof.merge(meta[["Metadata_JCP2022", "Metadata_NCBI_Gene_ID"]], on="Metadata_JCP2022")
    ent = dict(zip(h.entrez_id.dropna().astype(int), h.ensembl_gene_id[h.entrez_id.notna()],
                   strict=False))
    prof["ensg"] = prof.Metadata_NCBI_Gene_ID.map(
        lambda v: ent.get(int(float(v))) if str(v).replace(".0", "").isdigit() else None)
    prof = prof[prof.ensg.notna()]
    df = prof.groupby("ensg")[feats].mean()
    x = standardise(df.to_numpy())
    return list(df.index), (pca(x) if x.shape[1] > DIM else x)


def string(h: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    import scipy.sparse as sp
    from scipy.sparse.linalg import eigsh
    info = pd.read_csv(RAW / "string_info.txt.gz", sep="\t")
    info.columns = [c.lstrip("#") for c in info.columns]
    s2e = symbol_to_ensg(h)
    p2e = {p: s2e.get(str(n).upper()) for p, n in zip(info.string_protein_id, info.preferred_name,
                                                      strict=False)}
    links = pd.read_csv(RAW / "string_links.txt.gz", sep=" ")
    links = links[links.combined_score >= 400]
    a = links.protein1.map(p2e)
    b = links.protein2.map(p2e)
    ok = a.notna() & b.notna()
    genes = sorted(set(a[ok]) | set(b[ok]))
    ix = {g: i for i, g in enumerate(genes)}
    r = a[ok].map(ix).to_numpy()
    c = b[ok].map(ix).to_numpy()
    w = links.combined_score[ok].to_numpy() / 1000.0
    m = sp.coo_matrix((w, (r, c)), shape=(len(genes), len(genes))).tocsr()
    m = m.maximum(m.T)
    d = np.asarray(m.sum(1)).ravel()
    dinv = sp.diags(1.0 / np.sqrt(np.maximum(d, 1e-9)))
    norm = dinv @ m @ dinv
    vals, vecs = eigsh(norm, k=DIM + 1, which="LA")
    order = np.argsort(-vals)[1:]                       # drop the trivial top eigenvector
    return genes, (vecs[:, order] * vals[order]).astype(np.float32)


def esm2(h: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    import torch
    from transformers import AutoModel, AutoTokenizer
    u2e = {}
    for us, e in zip(h.uniprot_ids, h.ensembl_gene_id, strict=False):
        if isinstance(us, str) and isinstance(e, str):
            for u in us.split("|"):
                u2e[u] = e
    seqs = {}
    with gzip.open(RAW / "uniprot_human.fasta.gz", "rt") as fh:
        acc, buf = None, []
        for line in fh:
            if line.startswith(">"):
                if acc and acc in u2e and u2e[acc] not in seqs:
                    seqs[u2e[acc]] = "".join(buf)
                acc = line.split("|")[1]
                buf = []
            else:
                buf.append(line.strip())
        if acc and acc in u2e and u2e[acc] not in seqs:
            seqs[u2e[acc]] = "".join(buf)
    name, rev = "facebook/esm2_t33_650M_UR50D", "main"
    tok = AutoTokenizer.from_pretrained(name, revision=rev)
    model = AutoModel.from_pretrained(name, revision=rev, dtype=torch.float16).eval().cuda()
    genes = sorted(seqs, key=lambda g: len(seqs[g]))
    out = []
    with torch.no_grad():
        for i in range(0, len(genes), 32):
            batch = [re.sub(r"[^A-Z]", "", seqs[g])[:1022] for g in genes[i:i + 32]]
            t = tok(batch, return_tensors="pt", padding=True).to("cuda")
            hdn = model(**t).last_hidden_state.float()
            mask = t["attention_mask"][..., None].float()
            out.append(((hdn * mask).sum(1) / mask.sum(1)).cpu().numpy())
    return genes, pca(standardise(np.concatenate(out)))


def main() -> int:
    which = sys.argv[1].split(",") if len(sys.argv) > 1 else [
        "perturbseq", "jump_crispr", "jump_orf", "string", "esm2"]
    h = hgnc()
    OUT.mkdir(parents=True, exist_ok=True)
    fns = {"perturbseq": perturbseq, "jump_crispr": lambda: jump("crispr", h),
           "jump_orf": lambda: jump("orf", h), "string": lambda: string(h),
           "esm2": lambda: esm2(h)}
    for src in which:
        genes, x = fns[src]()
        np.savez(OUT / f"{src}.npz", genes=np.array(genes), X=x)
        print(f"{src}: {len(genes)} genes x {x.shape[1]}")
    # coverage of MitoCheck genes (by current Ensembl ID)
    split = pd.read_csv(REPO / "splits" / "gene_split.tsv", sep="\t")
    ann = pd.read_csv(REPO / "data" / "annotations" / "idr0013-screenA-annotation.csv.gz",
                      usecols=["Gene Symbol", "Gene Identifier"], low_memory=False)
    s2e = symbol_to_ensg(h)
    g2e = {}
    for s, e in ann.drop_duplicates().itertuples(index=False):
        if isinstance(s, str):
            cur = set(h.ensembl_gene_id.dropna())
            g2e[s] = e if isinstance(e, str) and e in cur else s2e.get(s.upper(), e)
    pd.Series(g2e).rename_axis("gene").rename("ensembl_gene_id").to_csv(
        OUT / "mitocheck_gene_to_ensg.tsv", sep="\t")
    cov = {}
    for src in sorted(p.stem for p in OUT.glob("*.npz")):
        have = set(np.load(OUT / f"{src}.npz")["genes"])
        cov[src] = {sp: int(sum(g2e.get(g) in have for g in split.gene[split.split == sp]))
                    for sp in ("train", "val", "test")}
        cov[src]["total_genes_in_source"] = len(have)
    cov["mitocheck_genes"] = {sp: int((split.split == sp).sum()) for sp in ("train", "val", "test")}
    (REPO / "results").mkdir(exist_ok=True)
    (REPO / "results" / "actions_coverage.json").write_text(json.dumps(cov, indent=1))
    print(json.dumps(cov, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
