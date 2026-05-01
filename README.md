# 🔍 Sentiment Intelligence Platform
### Beyond Star Ratings — Aspect-Level Product Intelligence

> An end-to-end NLP pipeline that transforms raw Amazon product reviews into structured, actionable product intelligence powered by aspect-based sentiment analysis, FAISS similarity search, and a LangChain ReAct agent.

🌐 **Live Demo**: [sentiment-intelligence-platform-textmining.streamlit.app](https://sentiment-intelligence-platform-textmining.streamlit.app)  
📦 **Dataset**: [Amazon Product Reviews — Kaggle](https://www.kaggle.com/datasets/manasabollavarapu/amazon-product-reviews/data)

---

## 💡 The Problem

A product rated 3★ might have **exceptional taste but terrible packaging**. A 1★ review might be entirely about a delayed shipment, not the product itself. A single star rating collapses all of this nuance into one useless number.

**This platform solves that.** We extract what customers are actually talking about taste, price, packaging, nutrition — and score each dimension separately. Every product becomes a **13-dimensional sentiment vector** that powers smarter search, gap-fill recommendations, and AI-generated explanations grounded in real reviews.

---

## 🚀 Live Features

| Feature | Description |
|---------|-------------|
| 🤖 **AI Product Advisor** | Chat with a ReAct agent powered by Llama 3.3 70B — asks follow-up questions, compares products, fetches real review evidence |
| 🔍 **Product Explorer** | Visualize any product's sentiment across 13 dimensions with strength/weakness badges |
| 🔗 **Similar Products** | FAISS cosine similarity search across 691 product vectors |
| 🎯 **Gap-Fill Recommender** | Find alternatives that score better specifically in a product's weak aspects |
| 📖 **About** | Project overview, tech stack, and team |

---

## 🧠 Pipeline Architecture
```
500,000+ Amazon Reviews
        │
        ▼
┌─────────────────────────────────┐
│  Stage 1 — Classical Baselines  │
│  VADER · TF-IDF · LogReg · XGB  │
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│  Stage 2 — Transformer Models   │
│  Zero-shot RoBERTa              │
│  Fine-tuned DistilBERT          │
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│  Stage 3 — ABSA Pipeline        │
│  spaCy noun chunks              │
│  PyABSA triplet extraction      │
│  76K aspects → 717 → 13 labels  │
│  Per-product 13-dim vectors     │
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│  Stage 4 — Recommendation       │
│  FAISS IndexFlatIP              │
│  Gap-fill recommender           │
│  ChromaDB (4,963 embeddings)    │
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│  Stage 5 — Agentic LLM Layer    │
│  LangChain ReAct Agent          │
│  Llama 3.3 70B via Groq         │
│  5 custom tools                 │
│  Grounded explanations          │
└─────────────────────────────────┘
```

---

## 📊 Dataset

| Field | Description | Role |
|-------|-------------|------|
| `Text` | Full review body | Primary NLP input |
| `Score` | Star rating (1–5) | Pseudo-label for training |
| `ProductId` | Product identifier | Grouping key for vectors |
| `UserId` | User identifier | User profile construction |
| `Summary` | Review headline | Supplementary signal |

**Label Construction:**
1–2 stars  →  Negative
3 stars    →  Neutral
4–5 stars  →  Positive

---

## 🛠️ Tech Stack

| Layer | Tools |
|-------|-------|
| Data & ETL | pandas, PySpark, pyarrow |
| Classical NLP | VADER, TF-IDF, Logistic Regression, XGBoost |
| Deep NLP | HuggingFace Transformers, DistilBERT, RoBERTa |
| ABSA | PyABSA, spaCy, Sentence-Transformers, KMeans |
| Recommendation | FAISS, ChromaDB, numpy |
| Agentic LLM | LangChain, Llama 3.3 70B, Groq |
| Experiment Tracking | MLflow |
| Frontend | Streamlit |
| Storage | Git LFS |
| Deployment | Streamlit Cloud |

---

## 📁 Project Structure
```
Sentiment-Intelligence-Platform/
│
├── app.py                          # Streamlit dashboard (main entry point)
├── requirements.txt                # Python dependencies
├── .python-version                 # Python 3.11 (for Streamlit Cloud)
├── README.md
│
├── notebooks/
│   ├── stage3_normalization.ipynb  # ABSA + KMeans taxonomy normalization
│   ├── stage4_recommendation.ipynb # FAISS + ChromaDB pipeline
│   └── stage5_agentic_llm.ipynb   # LangChain ReAct agent
│
├── src/
│   ├── stage1_classical.py         # VADER, TF-IDF, LogReg, XGBoost
│   ├── stage2_transformers.py      # RoBERTa zero-shot + DistilBERT fine-tuning
│   ├── stage3_absa.py              # ABSA extraction pipeline
│   ├── preprocess.py               # Data loading and cleaning
│   ├── config.py                   # All settings and hyperparameters
│   ├── main.py                     # Pipeline entry point
│   └── utils.py                    # Shared helpers
│
└── models/
    ├── absa/
    │   ├── product_aspect_vectors.csv   # 691 × 13 sentiment vectors
    │   ├── aspect_taxonomy_map.json     # raw aspect → taxonomy mapping
    │   └── product_names.csv            # ProductId → display name
    └── stage4/
        └── chroma_db/                   # ChromaDB review embeddings
---

## ⚙️ Local Setup

```bash
# 1. Clone the repo
git clone https://github.com/harshpatel8343/Sentiment-Intelligence-Platform.git
cd Sentiment-Intelligence-Platform

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set environment variables
# Create a .env file:
echo "GROQ_API_KEY=your_key_here" > .env
echo "APP_PASSWORD=your_password_here" >> .env

# 4. Run the app
streamlit run app.py
```

Get a free Groq API key (no credit card) at [console.groq.com](https://console.groq.com)

---

## 📈 Key Results

| Stage | Output |
|-------|--------|
| Stage 1 Baselines | Macro F1: LogReg ~0.77, XGBoost ~0.78 |
| Stage 2 Transformers | Fine-tuned DistilBERT Macro F1: ~0.83 |
| Stage 3 ABSA | 76K raw aspects → 717 cleaned → 13 taxonomy labels |
| Stage 4 Vectors | 691 products × 13 sentiment dimensions |
| Stage 5 Agent | Grounded recommendations with real review evidence |

---

## 🎯 Key Design Decisions

**Why aspect-level?** Star ratings are misleading — a 3★ product might be excellent on taste but poor on packaging. Aspect-level sentiment captures this nuance.

**Why KMeans for taxonomy?** We clustered 717 cleaned aspects using Sentence-Transformers embeddings. This let us normalize noisy extracted phrases into interpretable labels without manual labeling of all 717 terms.

**Why gap-fill recommender?** Standard similarity search finds products that are similar overall. Gap-fill specifically finds products that are better in the exact dimensions where the current product falls short — much more useful for users.

**Why ReAct agent?** The agent autonomously chains tool calls — searching by keyword, comparing aspect profiles, fetching real reviews — without being told which tools to use. This makes responses grounded and explainable.

---

## 👥 Team

| Name | NetID | Institution |
|------|-------|-------------|
| Harsh Patel | harshnp2 | UIUC |
| Aman Joshi | abj4 | UIUC |
| Shithil Shetty | shetty7 | UIUC |
| Kanishka Gupta | kgupta17 | UIUC |

---

## 📚 Course

**IS567 — Text Mining** · Spring 2026  
University of Illinois Urbana-Champaign

---

## 📄 License

This project is for academic purposes only.