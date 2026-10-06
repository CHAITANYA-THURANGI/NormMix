# 🌐 Complete Step-by-Step Guide: NormMix Online Publishing & Deployment

This guide documents the **live online deployment** of the **NormMix AI** Telugu-English Code-Mixed Normalization and Translation platform. Everything is hosted on **100% free, permanent tiers** designed for students and researchers.

---

## 🌟 Live Production Ecosystem

| Component | Platform | Live URL / Target | Cost | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Web Studio (Frontend)** | **GitHub Pages** | [chaitanya-thurangi.github.io/NormMix](https://chaitanya-thurangi.github.io/NormMix/) | **100% Free** | 🟢 **LIVE & Active** |
| **Neural API (Backend)** | **Render Cloud** | [normmix-api.onrender.com](https://normmix-api.onrender.com) | **100% Free** | 🟢 **LIVE & Serving** |
| **Interactive API Docs** | **FastAPI Swagger** | [normmix-api.onrender.com/docs](https://normmix-api.onrender.com/docs) | **100% Free** | 🟢 **LIVE & Interactive** |
| **Telemetry & Health** | **Render Health** | [normmix-api.onrender.com/health](https://normmix-api.onrender.com/health) | **100% Free** | 🟢 **200 OK (CPU)** |
| **Chrome Extension Package** | **GitHub Releases** | [NormMix v0.5.0 Release](https://github.com/CHAITANYA-THURANGI/NormMix/releases/tag/v0.5.0) | **100% Free** | 🟢 **Downloadable ZIP** |
| **Official Chrome Web Store** | **Google Dev Console** | Optional Future Listing | $5 one-time | Optional (Requires $5 fee) |

> [!TIP]
> **Zero Downtime Hybrid Dual-Engine Architecture:**
> The web studio and browser extension are connected to `https://normmix-api.onrender.com`. If the free cloud server takes >5 seconds to spin up from cold sleep, the frontend seamlessly and transparently falls back to the **In-Browser SOTA Client Engine (0ms latency)**. Users never see an error!

---

## 🚀 PART 1: The Website on GitHub Pages (100% Free)

### Current Live Status:
- **Repository:** [`https://github.com/CHAITANYA-THURANGI/NormMix`](https://github.com/CHAITANYA-THURANGI/NormMix)
- **Live URL:** [`https://chaitanya-thurangi.github.io/NormMix/`](https://chaitanya-thurangi.github.io/NormMix/)
- **Direct Web Studio:** [`https://chaitanya-thurangi.github.io/NormMix/web/index.html`](https://chaitanya-thurangi.github.io/NormMix/web/index.html)

### How It Works:
1. Root [`index.html`](file:///c:/projects/NormMix/index.html) serves an instant client redirect to `web/index.html`.
2. Static assets (CSS, JavaScript, Google Fonts, Telugu Lexicon Matrix) are served over global GitHub CDN with SSL/HTTPS.
3. The Web Studio features:
   - **Universal Modality Detection** (Pure English, Native Telugu Script, Tanglish, Bi-scriptal).
   - **4 Multi-Modal Targets**: Standard Normalized, All Telugu Script (అచ్చ తెలుగు), All Romanized Tanglish, and English Meaning.
   - **Google Input Tools Style Suggestions Dropdown** (`1. నాకు 2. నాకూ 3. నకు`).
   - **Audio Playback (TTS)** and **Microphone Speech Recognition**.
   - **Virtual Telugu Keyboard Drawer**.
   - **User Active Learning Loop** (saves feedback to `data/feedback.jsonl`).

---

## ⚡ PART 2: The Neural API Backend on Render.com (100% Free)

### Current Live Status:
- **Cloud Provider:** [Render.com](https://render.com) (Free Web Service Tier: 0.1 CPU, 512 MB RAM, $0/month)
- **Deployment Mode:** Docker Container
- **Live Endpoint:** `https://normmix-api.onrender.com`
- **Swagger Documentation:** `https://normmix-api.onrender.com/docs`
- **Health Check:** `https://normmix-api.onrender.com/health`

### Architecture & Container Optimizations:
1. **CPU-Optimized PyTorch Wheel:**
   Our production [`Dockerfile`](file:///c:/projects/NormMix/Dockerfile) downloads the lightweight CPU PyTorch wheel (~180 MB instead of 2.5 GB CUDA), fitting neatly within Render's free 512 MB RAM limit and building in under 2 minutes.
2. **Dynamic Port Binding:**
   The container starts with `CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-7860}"]`, automatically binding to Render's dynamic `$PORT` environment variable.
3. **Non-Root User (`UID 1000`):**
   Runs as an unprivileged user for modern container security standards.
4. **Permissive CORS:**
   Configured with `allow_origins=["*"]` to allow requests from GitHub Pages (`chaitanya-thurangi.github.io`) and Chrome Extensions (`chrome-extension://*`).

### How It Was Deployed (Reference):
1. Connected GitHub repository `CHAITANYA-THURANGI/NormMix` to Render.
2. Created a new **Web Service** $\to$ selected **Docker runtime**.
3. Render automatically reads `Dockerfile`, builds the image, and deploys it.
4. Every future `git push origin main` triggers an automatic rebuild and zero-downtime redeployment!

---

## 🧩 PART 3: Distribute the Chrome Extension

You have two paths to distribute your extension to users:

### Option A: Free GitHub Release Distribution (Active Now)
*Zero cost, immediate availability, no waiting for Google review.*

1. **Pre-Built Package:**
   The package [`normmix-chrome-extension-v0.5.0.zip`](file:///c:/projects/NormMix/normmix-chrome-extension-v0.5.0.zip) (11.7 KB) is already built and available in the root and `web/` directory.
2. **GitHub Release:**
   Published under tag **`v0.5.0`** at:
   [`https://github.com/CHAITANYA-THURANGI/NormMix/releases/tag/v0.5.0`](https://github.com/CHAITANYA-THURANGI/NormMix/releases/tag/v0.5.0)
3. **How Users Install It (Step-by-Step):**
   - Step 1: Download `normmix-chrome-extension-v0.5.0.zip`.
   - Step 2: Unzip the file to a folder on their computer.
   - Step 3: Open Chrome/Edge and go to `chrome://extensions/`.
   - Step 4: Toggle on **Developer mode** (top-right corner).
   - Step 5: Click **Load unpacked** $\to$ select the unzipped `chrome-extension` folder.
   - Step 6: The NormMix AI icon will appear in the browser toolbar!
4. **Web 1-Click Download:**
   Visitors on your live website can click the **Chrome Extension** button in the header navigation to directly download the ZIP.

---

### Option B: Official Chrome Web Store Listing ($5 USD One-Time Fee)
*Optional: If you wish to publish to the public Chrome Web Store search directory.*

1. **Register Developer Account:**
   Go to [Chrome Web Store Developer Dashboard](https://chrome.google.com/webstore/devconsole) and pay the one-time $5 USD registration fee.
2. **Upload Package:**
   Click **+ New Item** $\to$ drag and drop `normmix-chrome-extension-v0.5.0.zip`.
3. **Store Listing Details:**
   - **Name:** `NormMix AI - Universal Telugu-English Normalizer`
   - **Summary:** `Translate and normalize Telugu-English code-mixed (Tanglish) text with SOTA AI.`
   - **Category:** `Productivity` or `Search Tools`
   - **Language:** English
   - **Icons:** Automatically extracted from the package (`icons/icon16.png`, `icon48.png`, `icon128.png`).
4. **Permissions Justification:**
   - `storage`: Saves user preferences (offline mode, API selection, last input).
   - `activeTab`: Allows reading selected text when user right-clicks to normalize.
   - `contextMenus`: Adds the right-click "Normalize with NormMix" menu item.
5. **Submit for Review:**
   Review typically takes 24–48 hours, after which the extension gets a public store link.

---

## 🛠️ PART 4: Maintenance & Update Workflow

Whenever you edit files locally (such as HTML, Python models, or documentation), update both GitHub Pages and Render simultaneously with these simple commands:

```powershell
cd C:\projects\NormMix

# 1. Check which files were modified
git status

# 2. Stage all modifications
git add .

# 3. Commit with a meaningful message
git commit -m "feat: your change description here"

# 4. Push to GitHub
git push origin main
```

### What Happens Automatically After `git push`:
- **GitHub Pages:** Detects new files and updates your live website in ~45 seconds.
- **Render.com:** Detects the commit via webhook, triggers a fresh Docker build, and updates the live API in ~90 seconds.

---

## 🧪 Testing the Live System from PowerShell

You can verify the entire live ecosystem directly from your terminal:

```powershell
# 1. Test Live Cloud API Health
curl -s https://normmix-api.onrender.com/health

# 2. Test Live Code-Mixed Normalization
curl -X POST https://normmix-api.onrender.com/omni/process `
  -H "Content-Type: application/json" `
  -d '{"text": "naku ivala college lo important interview undi"}'

# 3. Test Live Web Studio Availability
curl -I https://chaitanya-thurangi.github.io/NormMix/web/index.html
```

All commands return `200 OK` with full multi-modal normalization output!
