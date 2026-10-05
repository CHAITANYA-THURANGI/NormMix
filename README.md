# NormMix AI: An Attention-Based Sequence-to-Sequence Framework for Telugu-English Code-Mixed Text Normalization & Universal Translation

[![Live Web Studio](https://img.shields.io/badge/Live%20Web%20Studio-Online%20(Free)-38bdf8?style=for-the-badge&logo=githubpages&logoColor=white)](https://chaitanya-thurangi.github.io/NormMix/)
[![Chrome Extension](https://img.shields.io/badge/Chrome%20Extension-v0.5.0%20Release-f59e0b?style=for-the-badge&logo=googlechrome&logoColor=white)](https://github.com/CHAITANYA-THURANGI/NormMix/releases/tag/v0.5.0)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11%2Bcu128-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-NVIDIA%20RTX%204060-76B900.svg?style=flat&logo=nvidia)](https://developer.nvidia.com/cuda-zone)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Tests](https://img.shields.io/badge/Tests-58%20Passed-10b981.svg?style=flat)]()
[![Documentation](https://img.shields.io/badge/Report%20PDF-804%20KB-blue.svg?style=flat)](docs/project_report.pdf)
[![Master Guide](https://img.shields.io/badge/Master%20Guide%20PDF-961%20KB-059669.svg?style=flat)](docs/PROJECT_COMPLETE_GUIDE_AND_LEARNING_PATH.pdf)

> **Course:** Advanced Artificial Intelligence and Neural Networks (AAN) Mini-Project & Case Study  
> **Architecture:** 2-Layer BiGRU + Scaled Dot-Product Attention + Pointer-Generator Copy Mechanism  
> **Hardware Acceleration:** NVIDIA GeForce RTX 4060 Laptop GPU (8.0 GB GDDR6)  


---

## 1. Project Overview

**NormMix AI** is an enterprise-grade NLP framework designed for South Asian digital communications where Telugu and English are heavily code-mixed (**Tanglish**). It provides:
1. **Universal Auto-Detection**: Automatically classifies text as *Pure English*, *Native Telugu Script*, *Romanized Tanglish*, or *Bi-scriptal Code-Mixed*.
2. **SOTA Code-Mixed Normalization**: Normalizes Romanized Telugu morphemes to pristine Telugu Unicode script while preserving English loanwords with 100% fidelity via a **Pointer-Generator Copy Gate**.
3. **Omni-Directional Multi-Modal Generation**: Produces 4 simultaneous target options for every input:
   - 🌟 **Standard Normalized**: Telugu in Telugu script, English loanwords preserved in Latin characters.
   - 🔤 **All Telugu Script**: Entire sentence converted to Telugu Unicode (loanwords phonetically transliterated).
   - 🔠 **All Romanized Tanglish**: Entire sentence phonetically transliterated to Latin characters.
   - 🌐 **English Semantic Meaning & Gloss**: Word-by-word bilingual vocabulary translation.
4. **Interactive Translation Studio (`web/index.html`)**:
   - Google Input Tools style **Real-time Phonetic Suggestions Dropdown** (`1. నాకు 2. నాకూ 3. నకు`).
   - **Text-to-Speech (TTS) Audio Playback** (`🔊 Listen`).
   - **Voice Input Microphone Dictation** (`🎙️ Speak`).
   - **Virtual Telugu On-Screen Keyboard Drawer** (`⌨️ Telugu Keyboard`).
   - **Active Learning User Feedback Loop** (`👍 / 👎` sending to `data/feedback.jsonl`).
   - **1-Click WhatsApp Share** and **Text File Download**.
5. **Cross-Platform Manifest V3 Chrome Extension (`chrome-extension/`)**:
   - Popup translator with auto-detect.
   - Context menu to normalize text on any webpage (WhatsApp Web, Twitter, LinkedIn, Gmail).

---

## 2. Key Benchmark Highlights

Evaluated across standard benchmarks and fine-grained test slices:

| Metric / Dimension | Baseline (Pre-Scaling) | **NormMix SOTA v2.0 (Current)** | Delta |
| :--- | :--- | :--- | :--- |
| **Zero-Leakage Training Pairs** | 1,862 pairs | **25,803 pairs** | **$13.8\times$ scale** |
| **Verified Lexicon Knowledge Base** | 1,023 entries | **64,421 entries** | **$63\times$ expansion** |
| **Model Architecture** | Vanilla Seq2Seq (CPU) | **2-Layer BiGRU + Copy Gate (RTX 4060 GPU)** | Pointer-Generator SOTA |
| **Gold Benchmark CER** | 0.4300 (43.0%) | **0.0537 (5.37%)** | **$8\times$ error reduction** |
| **Gold Benchmark chrF** | 56.40 | **91.29** | **+34.89 points** |
| **Gold Benchmark BLEU** | 35.10 | **79.84** | **+44.74 points** |
| **English Loanword Accuracy** | 22.0% (Corrupted) | **100.0% (Preserved)** | **Zero English degradation** |
| **Unseen Vocabulary BLEU** | 11.70 | **39.67** | **$3.4\times$ generalization** |
| **GPU Inference Latency** | ~28 ms | **~8.2 ms (CUDA FP16)** | Sub-10ms real-time |

---

## 3. Quickstart & How to Run

### Local Setup
```bash
# 1. Activate Python virtual environment
.\.venv\Scripts\activate   # On Windows
source .venv/bin/activate  # On Linux/macOS

# 2. Run the complete pytest test suite (58 tests)
python -m pytest

# 3. Start the FastAPI backend & Web Studio
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

### Accessing the System
- **Web Translation Studio:** [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- **Interactive Swagger Documentation:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Master Architecture & Learning Guide PDF:** [http://127.0.0.1:8000/guide.pdf](http://127.0.0.1:8000/guide.pdf)
- **Publication Research PDF Report:** [http://127.0.0.1:8000/report.pdf](http://127.0.0.1:8000/report.pdf)

---

## 4. Chrome Extension Installation

1. Open Google Chrome or Microsoft Edge and navigate to `chrome://extensions/`.
2. Enable **Developer mode** (toggle in the top-right corner).
3. Click **Load unpacked** and select the [`chrome-extension/`](file:///c:/projects/NormMix/chrome-extension) directory in this repository.
4. The extension icon will appear in your browser toolbar! Click it to open the popup or right-click any text on any webpage to normalize it instantly.

---

## 5. Repository Structure

```
├── Dockerfile                   # 1-click cloud deployment (Hugging Face Spaces, Render, Cloud Run)
├── requirements.txt             # Python dependencies
├── config.yaml                  # Global training and inference configurations
├── src/
│   ├── pipeline/                # OmniProcessor universal auto-detection & multi-modal pipeline
│   ├── preprocessing/           # Unicode cleaning, language ID, phonetic transliteration
│   ├── attention/               # Scaled Dot-Product, Bahdanau, Luong attention implementations
│   ├── models/                  # Pointer-Generator BiGRU, Transformer, Rule Baseline
│   ├── datasets/                # Zero-leakage splitting, Google Dakshina, AI4Bharat Aksharantar
│   └── inference/               # Beam search generation and GPU inference
├── api/
│   ├── main.py                  # FastAPI REST service (/omni/process, /normalize, /romanize, /feedback)
│   ├── schemas.py               # Pydantic request and response schemas
│   └── services/                # Model registry and rate limiting
├── web/
│   └── index.html               # Enterprise Web Translation Studio (Audio, Voice, Keyboard, Suggestions)
├── chrome-extension/            # Manifest V3 browser extension (Popup, Context Menu, Content Script)
├── docs/
│   ├── project_report.pdf       # Formal Academic & Industry Publication Report (804 KB, A4)
│   ├── project_report.html      # Academic source document
│   ├── 00_master_report.md      # Full Markdown technical report
│   └── deployment.md            # Cloud deployment guide (Docker, Cloud Run, Render)
├── data/
│   ├── processed/               # Zero-leakage train/val/test splits & rule baseline (64k entries)
│   └── feedback.jsonl           # Real-time user feedback and active learning logs
└── tests/                       # 47 automated PyTest unit tests
```

---

## 6. License
MIT License (see `LICENSE`). Datasets used: Google Dakshina (Apache-2.0), AI4Bharat Aksharantar (CC0).
