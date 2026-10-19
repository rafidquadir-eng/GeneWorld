"""Reference tables: complexes, the complex-held-out gene split, HeLa expression bins.

* Complexes: EBI Complex Portal human complexes (CC0; `complextab/9606.tsv`). CORUM (named in
  the design) was unreachable from this instance on 2026-10-19; Complex Portal is the curated
  substitute. UniProt members are mapped to MitoCheck gene symbols through HGNC.
* Complex split (design A.4): genes that share any complex are joined (union-find); whole
  components go to train / val / test (70 / 10 / 20 of genes, seeded); genes in no complex are
  singletons. Positive-control genes stay in train. Written to `splits/gene_split_complex.tsv`.
* HeLa expression: Human Protein Atlas cell-line RNA (CC BY-SA 4.0), HeLa nTPM, in 10
  quantile bins (bin 0 = not detected), for the expression-matched permutation null.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
REF = REPO / "data" / "ref"
SEED = 20261019


def main() -> int:
    h = pd.read_csv(REPO / "data" / "actions" / "raw" / "hgnc_complete_set.txt", sep="\t",
                    low_memory=False)
    u2s = {}
    for us, s in zip(h.uniprot_ids, h.symbol, strict=False):
        if isinstance(us, str):
            for u in us.split("|"):
                u2s[u] = s
    split = pd.read_csv(REPO / "splits" / "gene_split.tsv", sep="\t")
    g2e = pd.read_csv(REPO / "data" / "actions" / "mitocheck_gene_to_ensg.tsv", sep="\t")
    e2mc = {}
    for g, e in zip(g2e.gene, g2e.ensembl_gene_id, strict=False):
        e2mc.setdefault(e, g)
    s2e = dict(zip(h.symbol, h.ensembl_gene_id, strict=False))

    cp = pd.read_csv(REF / "complexportal_9606.tsv", sep="\t")
    rows = []
    for ac, name, mem in zip(cp["#Complex ac"], cp["Recommended name"],
                             cp["Identifiers (and stoichiometry) of molecules in complex"],
                             strict=False):
        for tok in str(mem).split("|"):
            u = re.sub(r"\(.*\)$", "", tok).split("-")[0]
            s = u2s.get(u)
            if s and s2e.get(s) in e2mc:
                rows.append((e2mc[s2e[s]], ac, name))
    comp = pd.DataFrame(rows, columns=["gene", "complex", "name"]).drop_duplicates()
    comp.to_csv(REF / "gene_complexes.tsv", sep="\t", index=False)

    # union-find over MitoCheck genes sharing a complex
    genes = list(split.gene)
    parent = {g: g for g in genes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for _, grp in comp.groupby("complex"):
        gs = [g for g in grp.gene if g in parent]
        for g in gs[1:]:
            parent[find(g)] = find(gs[0])
    comps: dict[str, list[str]] = {}
    for g in genes:
        comps.setdefault(find(g), []).append(g)
    ann = pd.read_csv(REPO / "data" / "annotations" / "idr0013-screenA-annotation.csv.gz",
                      usecols=["Control Type", "Gene Symbol"], low_memory=False)
    pos = set(ann.loc[ann["Control Type"] == "positive control", "Gene Symbol"].dropna())
    rng = np.random.default_rng(SEED)
    keys = sorted(comps)
    rng.shuffle(keys)
    n = len(genes)
    out, acc = {}, {"test": 0, "val": 0}
    for k in keys:
        mem = comps[k]
        if any(g in pos for g in mem):
            sp = "train"
        elif acc["test"] < 0.2 * n:
            sp = "test"
        elif acc["val"] < 0.1 * n:
            sp = "val"
        else:
            sp = "train"
        if sp in acc:
            acc[sp] += len(mem)
        for g in mem:
            out[g] = sp
    cs = pd.DataFrame({"gene": list(out), "split": list(out.values()),
                       "component_size": [len(comps[find(g)]) for g in out]})
    cs.to_csv(REPO / "splits" / "gene_split_complex.tsv", sep="\t", index=False)

    hela = pd.read_csv(REF / "hela_rna.tsv", sep="\t")
    hela["gene"] = hela.Gene.map(e2mc)
    hela = hela.dropna(subset=["gene"]).groupby("gene").nTPM.max().rename("nTPM").to_frame()
    det = hela.nTPM > 1
    hela["bin"] = 0
    hela.loc[det, "bin"] = 1 + pd.qcut(hela.nTPM[det].rank(method="first"), 9, labels=False)
    hela.to_csv(REF / "hela_expression_bins.tsv", sep="\t")

    summ = {"complex_memberships": len(comp), "complexes": int(comp.complex.nunique()),
            "genes_in_complexes": int(comp.gene.nunique()),
            "complex_split": {k: int(v) for k, v in cs.split.value_counts().items()},
            "largest_component": int(cs.component_size.max()),
            "hela_genes": int(len(hela)), "hela_detected": int(det.sum())}
    (REPO / "manifests" / "reference_summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
