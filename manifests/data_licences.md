# Data sources and licences

| source | use | licence | where |
|---|---|---|---|
| MitoCheck screenA/B movies (IDR idr0013; BioImage Archive S-BIAD865) | world-model training and evaluation | CC0 | EBI S3 `bia-integrator-data/S-BIAD865` |
| MitoCheck annotations (IDR metadata, idr0013-neumann-mitocheck) | splits, phenotype classes, validated hits | CC0 (IDR) | github.com/IDR/idr-metadata |
| Replogle et al. 2022 K562 genome-wide Perturb-seq pseudobulk | gene encoding | CC BY 4.0 | figshare 20029387 (file 35773217) |
| JUMP-CP cpg0016 CRISPR and ORF assembled profiles (v1.0b) | gene encodings | CC0 | Cell Painting Gallery (public S3, no account) |
| JUMP-CP metadata (crispr.csv.gz, orf.csv.gz) | perturbation-to-gene mapping | CC0 | github.com/jump-cellpainting/datasets |
| STRING v12.0 human links and info | gene encoding | CC BY 4.0 | stringdb-downloads.org |
| UniProtKB/Swiss-Prot human reviewed proteome | ESM-2 inputs | CC BY 4.0 | rest.uniprot.org |
| HGNC complete set | gene identifier mapping | CC0 | genenames.org (copy from TiJepa) |
| EBI Complex Portal human complexes (complextab 9606.tsv) | complex-held-out split; same-complex H3 hits | CC0 (Complex Portal 2022, NAR) | ftp.ebi.ac.uk/pub/databases/intact/complex/current |
| Human Protein Atlas cell-line RNA (HeLa nTPM) | expression-matched permutation null (H3) | CC BY-SA 4.0 | proteinatlas.org/download/tsv/rna_celline.tsv.zip |

CORUM (named in the design for complexes) did not respond from this instance on 2026-10-19
(connection failure); Complex Portal is used instead.
