# Seven-Segment OCR Home Assistant Add-on

A custom Home Assistant OS add-on that provides a local HTTP API to extract seven-segment characters from an image.

## Setup
Local setup with uv:
```bash
uv lock
uv run scripts/train.py
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```
