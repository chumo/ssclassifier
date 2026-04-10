#!/usr/bin/env bash
set -e

echo "Starting Seven-Segment OCR API..."

# The webserver binds to 0.0.0.0 over port 8118
uvicorn app.main:app --host 0.0.0.0 --port 8118
