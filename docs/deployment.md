# NormMix Production Deployment Guide

This document covers the production cloud deployment, local development, containerization, and browser extension distribution for the **NormMix AI** Telugu-English Code-Mixed Normalization and Translation framework.

---

## 🌟 Live Production Ecosystem

| Component | Platform | Live URL / Target | Architecture |
| :--- | :--- | :--- | :--- |
| **Web Studio (Frontend)** | **GitHub Pages** | [chaitanya-thurangi.github.io/NormMix](https://chaitanya-thurangi.github.io/NormMix/) | Static HTML5/CSS3/Vanilla JS + Client-Side SOTA Fallback Engine |
| **Neural API (Backend)** | **Render Cloud** | [normmix-api.onrender.com](https://normmix-api.onrender.com) | FastAPI + PyTorch 2.11 CPU + Uvicorn via Docker Container |
| **Interactive Docs** | **FastAPI Swagger** | [normmix-api.onrender.com/docs](https://normmix-api.onrender.com/docs) | OpenAPI 3.0 Interactive Documentation |
| **Telemetry & Health** | **Render Health** | [normmix-api.onrender.com/health](https://normmix-api.onrender.com/health) | Continuous uptime and hardware monitoring |
| **Chrome Extension** | **GitHub Releases** | [NormMix v0.5.0 Release](https://github.com/CHAITANYA-THURANGI/NormMix/releases/tag/v0.5.0) | Manifest V3 Unpacked Extension ZIP (11.7 KB) |

---

## 1. Cloud Architecture & Failover Strategy

NormMix implements a **Hybrid Dual-Engine Architecture**:

```mermaid
graph TD
    User["Web Browser / Mobile Visitor"] --> GH["GitHub Pages (Static Host)<br/>https://chaitanya-thurangi.github.io/NormMix/"]
    GH -->|Primary API Request (5s timeout)| Render["Render Cloud Backend (Docker)<br/>https://normmix-api.onrender.com"]
    Render -->|Neural Inference| BiGRU["3.05M BiGRU Pointer-Generator + 64k Lexicon"]
    GH -.->|Cold Start / Offline Failover (0ms)| ClientEngine["In-Browser Client SOTA Engine<br/>(Native JS Rule & Lexicon Matrix)"]
```

1. **Primary Route:** The client sends an asynchronous HTTP POST request to `https://normmix-api.onrender.com/omni/process` with a 5-second `AbortSignal` timeout.
2. **Seamless Fallback:** If the cloud service is spinning up from cold sleep (>5s) or the client is offline, execution instantly and transparently switches to the **In-Browser Client Engine**. The user never experiences an error.
3. **No Mixed Content / CORS Errors:** Both GitHub Pages and Render operate over secure **HTTPS**. Render's CORS configuration explicitly allows `["*"]`, enabling requests from web browsers and Chrome Extensions.

---

## 2. Docker Containerization (Render & Cloud Run)

The production container is defined in [`Dockerfile`](file:///c:/projects/NormMix/Dockerfile):

```dockerfile
FROM python:3.11-slim

# Setup non-root user with UID 1000 (Security standard)
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

# Install lightweight system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git && rm -rf /var/lib/apt/lists/*

# Install CPU-optimized PyTorch (~180MB instead of 2.5GB CUDA)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user:user . /home/user/app

RUN mkdir -p data/processed experiments/checkpoints experiments/logs && \
    chown -R user:user /home/user/app

USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH PORT=7860 PYTHONUNBUFFERED=1

EXPOSE 7860 8000

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
```

### Key Optimizations:
- **CPU-Optimized Wheel:** Downloads the official CPU PyTorch wheel (~180 MB) instead of CUDA binaries (2.5 GB), reducing container build times from 8 minutes to under 2 minutes.
- **Dynamic Port Binding:** Automatically binds to `${PORT:-7860}`, enabling compatibility across Render (`PORT=10000`), Hugging Face Spaces (`PORT=7860`), and Google Cloud Run (`PORT=8080`).
- **Non-Root User:** Operates under unprivileged UID `1000` to prevent privilege escalation.

---

## 3. Local Development & GPU Training

```bash
# 1. Activate Virtual Environment
.\.venv\Scripts\activate   # Windows PowerShell
source .venv/bin/activate  # Linux / macOS

# 2. Run Test Suite (62 automated pytest checks)
python -m pytest

# 3. Launch Local FastAPI Server (CUDA GPU Acceleration)
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```
Local Swagger UI: `http://127.0.0.1:8000/docs`

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
