# Local datasets

This directory is intentionally data-free in Git. Runtime traces, camera
frames, videos, replay buffers, DeepAccident downloads, and Bench2Drive
collections remain on each workstation and must never contain credentials.

The active reproducible generators are committed under `scripts/`:

- `run_report_dreamer_native_collection_matrix.sh` collects native SimLingo
  transitions for the report-aligned Dreamer.
- `run_residual_dreamerv3_pipeline.sh` audits, trains, and gates the isolated
  residual DreamerV3/RSSM branch.
- `prepare_deepaccident_mini.sh` downloads and prepares the optional
  DeepAccident feasibility subset.

Raw datasets are not needed to launch the dashboard or run an already promoted
checkpoint. They are required only when reproducing training or audits.
