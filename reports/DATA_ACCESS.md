# Data access (measured 2026-10-19, before any movie was embedded)

MitoCheck screenA/B is public under CC0 (IDR idr0013; BioImage Archive S-BIAD865 OME-NGFF
conversion). No AWS account and no login were used.

## Layout found

- S-BIAD865 per-plate archives (`ftp.ebi.ac.uk/biostudies/fire/S-BIAD/865/.../*.ome.zarr.zip`):
  26–31 GB each, ~755,000 zip members per plate (one chunk per frame per level; chunk data
  deflate-compressed inside the zip on top of Blosc).
- The same plates as **unzipped OME-Zarr stores on EBI S3**:
  `https://uk1s3.embassy.ebi.ac.uk/bia-integrator-data/S-BIAD865/<uuid>/<uuid>.zarr`;
  **538 plates** (screenA + screenB), **200,995 wells**; plate name in `.zattrs`
  (`manifests/s3_plate_map.json`).
- Each well: `<row>/<col>/0`, axes (t, c, z, y, x), **93 frames** (30 min), uint8, one chunk per
  frame, Blosc-LZ4. **Pyramid levels:** 0 = 1024 x 1344, **1 = 512 x 672**, 2 = 256 x 336,
  3 = 128 x 168. Compressed size per well (plate LT0001_02): 50.5 / **14.3** / 3.8 / 1.1 MB for
  levels 0–3.
- **Chosen:** level 1 (2x downsampled): four 256 x 256 tiles from the central 512 x 512.

## Endpoint tests (level-1 chunks, ~150–210 KB each)

| endpoint | streams | throughput | errors |
|---|---|---|---|
| EBI HTTP, byte ranges into the plate zip (keep-alive) | 16 | 1.6 MB/s | 11 / 1,500 |
| | 32 | 2.3 MB/s | 478 / 1,500 |
| | 64 | 5.4 MB/s | 1,372 / 1,500 |
| | 128 | 3.9 MB/s | 1,430 / 1,500 |
| | 256 | 1.3 MB/s | 1,476 / 1,500 |
| EBI HTTP, 64 concurrent fresh connections on one range | 64 | — | 39 / 64 refused |
| **EBI S3 (bia-integrator-data), direct chunk GETs** | 64 | 18.3 MB/s | 0 |
| | **128** | **37.8 MB/s** | **0** |
| | 256 | 29.2 MB/s | 0 |
| Globus personal endpoint | — | not tested | needs an interactive login to the author's Globus account (no unattended route) |

EBI HTTP limits concurrent connections per client; it suits a few large transfers (31 MB/s for
64 MB ranges in an earlier test) but not per-frame reads. **EBI S3 at 128 streams is used.**
Stage 1 (~15,000 wells x ~18 MB at level 1, ~270 GB) is ~2 h of transfer at that rate.

## Handling

Stream-and-embed: each well's 93 frames are fetched into memory, embedded and dropped; no raw
movie is written to disk (the rolling buffer is the in-memory queue, far below the 300 GB cap).
Latents, annotations and splits stay on local disk and are backed up with SHA-256 to
`~/LambdaFilesystem/geneworld_backups/` (`reports/BACKUP_LEDGER.tsv`).

## Annotations

IDR metadata (`github.com/IDR/idr-metadata`, `idr0013-neumann-mitocheck/`): screenA annotation
(196,608 wells, 512 plates, 18,060 genes, 51,616 siRNAs; per-well automatic and manual
phenotype scores; reproducible phenotype terms; validated-mitotic-hit flag), screenB annotation
(10,752 wells, 1,127 genes). SHA-256 in `manifests/annotations_sha256.txt`.
