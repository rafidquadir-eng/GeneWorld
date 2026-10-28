# GeneWorld manuscript (draft)

- `main_anon.tex`: anonymised TMLR submission version; `main_biorxiv.tex`: named preprint
  version (TMLR `preprint` option). Both input `body.tex`, `numbers.tex` and `refs.bib`.
- `numbers.tex` is generated: `python scripts/91_paper_numbers.py` (every number comes from
  `results/`, `manifests/` or `splits/`; never edit it by hand).
- Style files: `style/` from github.com/JmlrOrg/tmlr-style-file (Apache-2.0,
  `style/LICENSE-tmlr-style`).
- Build (no system TeX needed): `tectonic -X compile --outdir <build dir> main_anon.tex`.
- Results and discussion are filled from the results files (pre-registered gates first, then the
  post hoc robustness controls). Nothing here is
  submitted or posted; the author does that after review.
