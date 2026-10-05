# NormMix Deployment Guide

This document covers the various deployment options for the Telugu-English code-mixed text normalization project, including local development, web application deployment, browser extension, mobile application, and model export strategies.

## 1. Local Development

### Setting up the virtual environment

It is recommended to use a virtual environment to manage dependencies:

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Installing dependencies

Install the required packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Running the API server

Start the FastAPI server using `uvicorn`:

```bash
uvicorn api.main:app --reload
```

The API will be available at `http://127.0.0.1:8000`. You can access the interactive API documentation at `http://127.0.0.1:8000/docs`.

### Testing the web interface

If the web interface is served statically, you can open `web/index.html` in your browser. Alternatively, if it is served by the FastAPI application, navigate to the appropriate route (e.g., `http://127.0.0.1:8000/`).

## 2. Web Application Deployment

### Architecture

The web application follows a client-server architecture:
**Frontend (`web/index.html`) → FastAPI (`api/main.py`) → Model Inference**

### Local deployment with uvicorn

For local testing, as mentioned above:
```bash
uvicorn api.main:app --reload
```

### Production deployment with gunicorn/uvicorn workers

For production, use `gunicorn` with `uvicorn` workers to handle multiple concurrent requests:

```bash
gunicorn api.main:app -w 4 -k uvicorn.workers.UvicornWorker
```
*(Adjust the number of workers `-w` based on your server's CPU cores).*

### Docker containerization

You can containerize the application using Docker.

**Sample `Dockerfile`:**

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["gunicorn", "api.main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8000"]
```

Build and run the container:
```bash
docker build -t normmix-api .
docker run -p 8000:8000 normmix-api
```

### Google Cloud Run deployment (free tier)

1. Build and push the Docker image to Google Container Registry (GCR) or Artifact Registry.
2. Deploy to Cloud Run:
```bash
gcloud run deploy normmix-api --image gcr.io/[PROJECT-ID]/normmix-api --platform managed --region us-central1 --allow-unauthenticated
```
This leverages the Google Cloud free tier for serverless deployment.

### Environment variables and configuration

Use a `.env` file or environment variables for configuration. Key variables might include:
- `MODEL_PATH`: Path to the trained model weights.
- `LOG_LEVEL`: Logging level (e.g., INFO, DEBUG).
- `ALLOWED_ORIGINS`: Allowed origins for CORS.

## 3. Chrome Extension

### Loading the extension in Chrome (developer mode)

1. Open Google Chrome and navigate to `chrome://extensions/`.
2. Enable **Developer mode** (toggle in the top right).
3. Click **Load unpacked** and select the directory containing the extension files (e.g., `extension/`).

### How it communicates with the local API

The extension uses the `fetch` API to send HTTP POST requests to the local or remote API endpoint (e.g., `http://127.0.0.1:8000/normalize`). Ensure the API server is running and CORS is configured to accept requests from the extension.

### Permissions and privacy

The `manifest.json` file should declare the necessary permissions:
- `activeTab`: To access the content of the current tab.
- `storage`: To save user preferences (e.g., custom API URL).
- Privacy: The extension sends text to the API. Ensure the API does not log user-specific data (see Security Considerations).

### Publishing to Chrome Web Store (future)

1. Create a developer account on the Chrome Web Store.
2. Zip the extension directory.
3. Upload the zip file, provide screenshots, description, and privacy policy.
4. Submit for review.

## 4. Mobile Application

### Flutter scaffold overview

The mobile application is built using Flutter, providing cross-platform support (iOS and Android) from a single codebase.
- `lib/main.dart`: Application entry point.
- `lib/screens/`: UI screens (e.g., home, settings).
- `lib/services/`: API communication logic.

### Building and running

Ensure Flutter is installed and a device/emulator is running.

```bash
cd mobile
flutter pub get
flutter run
```

### API integration

The app uses the `http` package to communicate with the FastAPI backend. Update the API URL in the settings or a configuration file to point to your deployed server.

### Future: on-device inference with ONNX Runtime Mobile

Future versions may integrate the ONNX model directly into the app using `onnxruntime_mobile` for offline inference, reducing latency and improving privacy.

## 5. Model Export

### Exporting PyTorch model to ONNX

Export the PyTorch model to ONNX format for cross-platform compatibility:

```python
import torch
# Assuming 'model' is your loaded PyTorch model and 'dummy_input' is prepared
torch.onnx.export(model, dummy_input, "model.onnx", opset_version=14,
                  input_names=['input'], output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
```

### Converting to TensorFlow.js (for browser inference)

For client-side inference in the web app or extension, convert the ONNX model to TensorFlow.js:
1. Convert ONNX to TensorFlow SavedModel (e.g., using `onnx-tf`).
2. Convert SavedModel to TF.js using `tensorflowjs_converter`.

### TFLite conversion (for mobile)

Convert the model to TensorFlow Lite for optimal mobile performance:
1. Convert ONNX to TensorFlow SavedModel.
2. Use `TFLiteConverter` to create a `.tflite` file.

### Performance considerations

- **Quantization:** Apply int8 quantization during export to reduce model size and improve inference speed on CPUs/mobile devices.
- **Batching:** Ensure dynamic batching is supported if exporting for API usage.

## 6. Security Considerations

### HTTPS for production

Always use HTTPS in production to encrypt data in transit. Cloud Run provides this automatically. If using a custom VPS, set up Let's Encrypt with Nginx or Traefik.

### Rate limiting

Implement rate limiting in FastAPI (e.g., using `slowapi`) to prevent abuse and denial-of-service attacks.

### Input validation and sanitization

Validate all input text. Ensure it meets length requirements and sanitize it to prevent injection attacks, although less critical for simple text inputs, it's a good practice.

### No logging of user text in production

To protect user privacy, configure the logging system to **not** log the actual text sent for normalization in production environments. Only log metadata (e.g., request duration, status code).

### CORS configuration

Configure Cross-Origin Resource Sharing (CORS) in FastAPI to only allow requests from trusted origins (your web app domain, extension ID, etc.).

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourwebapp.com", "chrome-extension://your-extension-id"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```
