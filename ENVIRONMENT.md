# Environment (pinned, from the stage-1/2 instance, 2026-10-23)

- Python 3.12.3; CUDA 12.8 wheels (`torch==2.11.0+cu128`); 1 x A100-40GB driver stack.
- `python3.12 -m venv ~/venv && ~/venv/bin/pip install -r requirements-lock.txt`
  (torch from the CUDA 12.8 index: `--extra-index-url https://download.pytorch.org/whl/cu128`).
- Model weights (pinned in `manifests/models.yaml`, SHA-256 verified at load by
  `geneworld.eval.provenance.verified_backbone_record`): V-JEPA 2 and DINOv2 Hugging Face
  snapshots are also on the persistent filesystem under `geneworld_backups/hf_cache/`; copy them
  to `~/.cache/huggingface/hub/` or download the pinned revisions. TiJepa-1:
  `geneworld_backups/checkpoints/backbones/tijepa1_T1_seed0.pt`.
- Data not in git (latents, caches, checkpoints, annotations, encodings, reference tables) is on
  the persistent filesystem under `geneworld_backups/`, listed with SHA-256 in
  `reports/BACKUP_LEDGER.tsv` (audit: `results/a2/backup_audit.json`).
