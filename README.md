# GeneWorld

An action-conditioned latent world model of genome-scale live-cell RNAi time-lapse
(MitoCheck), built on a frozen V-JEPA 2 video encoder: predict how a cell population's latent
state evolves under a gene knockdown, transfer zero-shot to unseen genes through gene
encodings from other modalities, and plan by searching over gene actions.

- Design: `docs/NEXT_PROJECT_DESIGN.md` (section A), amended by `docs/DESIGN_AMENDMENT_01.md`.
- Selection evidence: `docs/NEXT_PROJECT_SELECTION.md`.
- Rules carried over from TiJepa: `docs/tijepa_rules/RULES.md`.
- Reused code from TiJepa (rafidquadir-eng/TiJepa): provenance, claims checker, leakage
  fingerprints, statistics, forecasting controls, the V-JEPA 2 wrapper and the TiJepa model.

Private repository; nothing here is published.
