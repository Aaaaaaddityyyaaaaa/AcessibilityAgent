---
title: Agent_Space
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# 🤖 AcessibilityAgent

An end-to-end **accessibility-focused agentic AI system** that takes an image or a text prompt and returns a spoken audio response. Built with a custom-trained image captioning model, a LangGraph agent pipeline, Groq LLM, and gTTS — deployed as a FastAPI backend on HuggingFace Spaces with a Streamlit frontend.

---

## 🧠 How It Works

```
User Input (image or prompt)
        │
        ▼
   FastAPI /Agent
        │
        ▼
   LangGraph Router
   ┌────┴────┐
   ▼         ▼
Captioning  LLM Agent
Agent       (Groq LLaMA)
   └────┬────┘
        ▼
   TTS Agent (gTTS)
        │
        ▼
  Audio Response (mp3)
```

- **Image input** → routed to the Captioning Agent → generates a text caption → converted to audio
- **Text prompt** → routed to the LLM Agent (Groq LLaMA 3.3 70B) → generates a response → converted to audio

---

## 🏗️ Architecture

### Models
| Component | Details |
|---|---|
| Feature Extractor | EfficientNet-B7 (ImageNet pretrained, classifier head removed) |
| Caption Decoder | GPT-2 with a trainable projection layer (2560 → 768) |
| LLM | LLaMA 3.3 70B via Groq API |
| TTS | Google Text-to-Speech (gTTS) |

### Training
- Dataset: Flickr8k (1000 images subset)
- Optimizer: AdamW with separate LRs for projection (1e-3) and GPT-2 (5e-5)
- Image preprocessing: 600×600 resize, ImageNet normalization
- Models tracked with **DVC** and stored on **DagsHub**
- Experiments tracked with **MLflow**

---

## 📁 Project Structure

```
AcessibilityAgent/
├── api/
│   └── api.py                  # FastAPI app with /Agent endpoint
├── src/
│   ├── captioninng_model/
│   │   ├── get_encoder.py      # EfficientNet-B7 feature extractor
│   │   ├── get_gpt2_decoder.py # GPT-2 decoder model
│   │   └── train.py            # Training loop
│   ├── llm/
│   │   ├── AgentGraph.py       # LangGraph pipeline
│   │   ├── Router.py           # Image vs prompt router
│   │   ├── captioning_agent.py # Captioning node
│   │   ├── llm_agent.py        # LLM node
│   │   ├── tts_agent.py        # TTS node
│   │   ├── state.py            # Shared state schema
│   │   └── wrapper.py          # AgentWrapper class
│   └── utils/
│       └── common.py           # YAML config reader
├── data/
│   ├── flickr_dataset.py       # Flickr8k PyTorch Dataset
│   ├── image_preprocess.py     # Preprocessing for files and base64
│   └── get_data.py             # KaggleHub dataset downloader
├── ui/
│   └── streamlit_app.py        # Streamlit frontend
├── model/feature_model/        # f_model.pkl (DVC tracked)
├── artifacts/model/            # decoder.pkl (DVC tracked)
├── config.yaml                 # All paths and hyperparameters
├── Dockerfile                  # Docker build for HF Spaces
├── docker-compose.yaml         # Local dev (backend + frontend)
└── requirements.txt
```

---

## 🚀 Deployment

### HuggingFace Spaces (Docker)
The backend is deployed as a Docker Space at:
```
https://aaaadddittyyaaa-agent-space.hf.space
```

Set these secrets in your HF Space settings:
| Secret | Description |
|---|---|
| `DAGSHUB_USERNAME` | Your DagsHub username |
| `DAGSHUB_TOKEN` | Your DagsHub access token |
| `GROQ_API_KEY` | Your Groq API key |

### Local Development
```bash
# Clone the repo
git clone https://github.com/Aaaaaaddityyyaaaaa/AcessibilityAgent.git
cd AcessibilityAgent

# Set up environment
cp .env.example .env  # fill in your keys

# Pull models from DagsHub
dvc pull

# Run with Docker Compose
docker-compose up
```

- Backend: `http://localhost:8000`
- Frontend: `http://localhost:8501`

---

## 📡 API

### `POST /Agent`

**Request body:**
```json
{
  "image": "<base64 encoded image string>",  // optional
  "prompt": "your question here"             // optional
}
```

**Response:**
```json
{
  "text": "generated caption or answer",
  "audio": "<base64 encoded mp3>"
}
```

**Example (Python):**
```python
import requests, base64

with open("image.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

res = requests.post(
    "https://aaaadddittyyaaa-agent-space.hf.space/Agent",
    json={"image": image_b64, "prompt": ""}
)
data = res.json()
print(data["text"])
```

---

## 🖥️ Frontend

A Streamlit UI is included at `ui/streamlit_app.py` with two modes:

- **Caption Image** — upload an image, get a caption + audio playback
- **Ask Question** — type a prompt, get an LLM answer + audio playback

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI + Uvicorn |
| Agent Pipeline | LangGraph |
| LLM | Groq (LLaMA 3.3 70B) |
| Vision | EfficientNet-B7 + GPT-2 |
| TTS | gTTS |
| Frontend | Streamlit |
| Model Registry | DVC + DagsHub |
| Experiment Tracking | MLflow |
| Deployment | HuggingFace Spaces (Docker) |

---

## 📊 Model Training

To retrain from scratch:

```bash
# 1. Download dataset
python data/get_data.py

# 2. Create feature extractor
python src/captioninng_model/get_encoder.py

# 3. Train decoder
python src/captioninng_model/train.py

# 4. Push models to DagsHub
dvc push
```

---

## 📄 License

MIT