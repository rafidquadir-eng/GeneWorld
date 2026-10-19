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

### Second round (2026-10-19, 22:50–23:20 UTC; curl 8.5 `--parallel`, level-1 chunks)

| endpoint | streams | throughput | errors |
|---|---|---|---|
| EBI S3, BIA bucket (`bia-integrator-data/S-BIAD865`), HTTPS | 128 | 91.0 MB/s | 0 / 3,720 |
| | 256 | 82.8 MB/s | 0 / 3,720 |
| | 128, fresh (uncached) wells | **85.0 MB/s** | 0 / 3,720 |
| EBI S3, IDR bucket (`idr/zarr/v0.4/idr0013A`, same data, axes t/y/x), HTTPS | 128 | 82.5 MB/s | 0 / 3,720 |
| | 256 | 91.6 MB/s | 0 / 3,720 |
| **BIA + IDR buckets at once** (2 x 128), fresh wells | 256 | **180.8 MB/s** | 0 / 7,440 |
| EBI S3 over plain HTTP (port 80) | — | unreachable | connection times out from this instance |
| EBI FTP (`ftp.ebi.ac.uk`), 128–256 MB byte ranges of a plate zip | 1 | 2.0 MB/s | — |
| | 16 | 11.1 MB/s | 8 / 16 connections refused |

The first-round S3 figure (37.8 MB/s) was limited by the Python thread-pool client, not the
server: the same endpoint gives 85–91 MB/s with curl's native parallel transfers, and the two
S3 buckets add up (~181 MB/s), so they are served independently. Stage 1 is GPU-bound
(0.55 wells/s needs ~10 MB/s), so it keeps the Python reader; for a stage-2 expansion the
fetcher should move to a native parallel client and split plates across both buckets
(IDR plate IDs map to plate names through each store's `.zattrs`).

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
