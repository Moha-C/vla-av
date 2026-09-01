# Streamlit read-only presentation

This app publishes a sanitized snapshot of the local SimLingo dashboard. It has
no CARLA, SUMO, process-control, replay, attack, or checkpoint-write capability.
The committed `kpi_snapshot.json` contains the same sanitized KPI payload as the
embedded dashboard, making the published numbers independently inspectable.

Refresh the KPI snapshot locally before pushing:

```bash
cd ~/Desktop/vla-av
python3 scripts/export_streamlit_dashboard.py
```

This refreshes both `dashboard_snapshot.html` and `kpi_snapshot.json` from the
current local Bench2Drive results. No log path or workstation path is exported.

Test it locally:

```bash
bash scripts/run_streamlit_dashboard_readonly.sh
```

For Streamlit Community Cloud, choose this entrypoint:

```text
streamlit_share/app.py
```

When the GitHub repository is private, keep the Streamlit app private and invite
the intended viewer by email from the app sharing settings.
