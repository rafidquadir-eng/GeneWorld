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
