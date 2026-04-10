# Seven-Segment OCR Home Assistant Add-on

A custom Home Assistant OS add-on that provides a local HTTP API to extract seven-segment characters from an image. It uses a Random Forest classifier combined with a robust geometry-based feature extraction pipeline.

## Features
- **Inference API**: Fast OCR inference via HTTP POST. Send an image path and a coordinate array to extract digit values.
- **Inference UI**: A web-based visual interface hosted at the root (`/`) to test digit detection by easily drawing coordinate boxes over images.
- **Training API**: Model training logic to learn new digits and specific configurations dynamically.
- **Training UI**: A web-based visual tool located at `/trainer` for uploading images, selecting regions by drawing coordinates, assigning digit labels, and building a training dataset to train models from scratch.

## Setup
Local setup managed with `uv`:
```bash
uv lock
uv run scripts/train.py  # Run initial training if you don't already have `models/rf_model.joblib`
uv run uvicorn app.main:app --host 0.0.0.0 --port 8118
```

## API Endpoints

### UI Routes
- `GET /`: Visual digit detection UI. Automatically scales images to fit your screen.
- `GET /trainer`: Visual UI for building Random Forest training datasets.

### Core Endpoints
- `POST /detect`: The primary OCR inference endpoint. Read coordinates and apply the model to return digits.
  - **Body Format**: `{"image_path": "/absolute/path/to/image.jpg", "coords": [p_x, p_y, x_x, x_y, y_x, y_y, ...]}`
  - **Returns**: `{"result": "123.4"}`
- `POST /upload`: Upload an image file for use in the Training UI. (Data saved to `app/static/uploads`)
- `POST /train`: Trains a new model to `models/rf_model.joblib` based on given annotated samples.
  - **Body Format**: 
    ```json
    {
      "samples": [
        {
          "image_path": "/absolute/path/to/image.jpg", 
          "coords": [p_x, p_y, x_x, x_y, y_x, y_y], 
          "label": "1"
        }
      ]
    }
    ```

## Home Assistant Integration
Since this runs as an isolated API, you can utilize Node-RED or standard REST integrations in Home Assistant to periodically trigger `/detect` with local camera frames, then publish the results as numeric sensor states via MQTT or webhook.
