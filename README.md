# 🧭 SentimentCompass

> **Understanding User Satisfaction, Topic Trends, and Migration Signals Across ChatGPT, Gemini, Claude, Microsoft Copilot, and Perplexity**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?style=flat-square&logo=streamlit)](https://streamlit.io)
[![Plotly](https://img.shields.io/badge/Plotly-5.18%2B-3F4F75?style=flat-square&logo=plotly)](https://plotly.com)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?style=flat-square)](https://huggingface.co)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

---

## 📌 Project Overview

SentimentCompass is a **full-stack NLP analytics project** built as a final-year B.Tech CSE (Data Science) internship project. It analyses **50,000 user reviews** of five leading Generative AI platforms using a multi-model sentiment pipeline, LDA + BERTopic topic modeling, and a rule-based migration detection system — all visualised through an interactive Streamlit dashboard.

### What it answers

| Research Question | Method |
|---|---|
| Which platform has the highest user satisfaction? | TWSI + multi-model sentiment |
| What are users actually complaining about? | LDA + BERTopic topic modeling |
| Which platform do users switch to most often? | Rule-based migration detection |
| What drives users to switch platforms? | 10-category reason extraction |
| Does community endorsement change the satisfaction picture? | Thumbs-Weighted Sentiment Index |

---

## 🗂️ Project Structure

```
sentimentcompass/
│
├── data/
│   ├── raw/
│   │   └── The__Generative_AI_Ecosystem_50k_User_Reviews_2026.csv
│   ├── processed/
│   │   ├── reviews_clean.parquet          ← Notebook 02 output
│   │   ├── reviews_sentiment.parquet      ← Notebook 03 output
│   │   ├── reviews_topics.parquet         ← Notebook 04 output
│   │   └── reviews_migration.parquet      ← Notebook 05 output (dashboard input)
│   └── external/                          ← Scraped supplement data
│
├── notebooks/
│   ├── 01_eda.ipynb                       ← EDA + Data Quality Audit
│   ├── 02_preprocessing.ipynb             ← Cleaning + Feature Engineering
│   ├── 03_sentiment_analysis.ipynb        ← VADER + TextBlob + RoBERTa + TWSI
│   ├── 04_topic_modeling.ipynb            ← LDA + BERTopic
│   └── 05_migration_analysis.ipynb        ← Competitive Intelligence
│
├── dashboard/
│   ├── app.py                             ← Streamlit home page
│   ├── pages/
│   │   ├── 1_Overview.py
│   │   ├── 2_Sentiment.py
│   │   ├── 3_Topics.py
│   │   ├── 4_Migration.py
│   │   └── 5_Data_Explorer.py
│   └── utils/
│       ├── data_loader.py
│       ├── helpers.py
│       └── charts.py
│
├── outputs/
│   └── figures/                           ← 40+ saved charts (300 DPI PNG)
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🚀 Quick Start

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/sentimentcompass.git
cd sentimentcompass
pip install -r requirements.txt
```

### 2. Place the dataset

Download the dataset and place it at:
```
data/raw/The__Generative_AI_Ecosystem_50k_User_Reviews_2026.csv
```

### 3. Run notebooks in order

```bash
jupyter notebook
```

Run each notebook top-to-bottom:

| Order | Notebook | Time (CPU) | Output |
|---|---|---|---|
| 1 | `01_eda.ipynb` | ~3 min | 14 figures, quality audit |
| 2 | `02_preprocessing.ipynb` | ~5 min | `reviews_clean.parquet` |
| 3 | `03_sentiment_analysis.ipynb` | ~35 min* | `reviews_sentiment.parquet` |
| 4 | `04_topic_modeling.ipynb` | ~30 min | `reviews_topics.parquet` |
| 5 | `05_migration_analysis.ipynb` | ~12 min | `reviews_migration.parquet` |

*RoBERTa runs on a 10K sample in CPU mode. Use Google Colab GPU for full 50K (~15 min).

### 4. Launch the dashboard

```bash
streamlit run dashboard/app.py
```

Opens at **http://localhost:8501**

---

## 📊 Dashboard Pages

| Page | Description |
|---|---|
| 🏠 **Home** | Project overview, platform cards, pipeline summary |
| 📊 **Overview** | KPI metrics, star ratings, sentiment stacked bars, platform scatter |
| 🧠 **Sentiment** | VADER · TextBlob · RoBERTa · TWSI · model correlation heatmap |
| 🔍 **Topics** | LDA frequency · platform-topic heatmap · pos/neg topic ranking |
| 🔄 **Migration** | NetworkX graph · migration matrix · competitive scorecards |
| 🗄️ **Data Explorer** | Searchable table · multi-filter · CSV export |

---

## 🛠️ Technology Stack

### NLP & Machine Learning
- **VADER** — rule-based sentiment (50K reviews, ~45 sec)
- **TextBlob** — polarity + subjectivity scoring
- **RoBERTa** — `cardiffnlp/twitter-roberta-base-sentiment-latest` (HuggingFace)
- **LDA** — `gensim.LdaMulticore` with coherence-optimised k
- **BERTopic** — semantic clustering with `all-MiniLM-L6-v2` embeddings

### Analytics
- **TWSI** — Thumbs-Weighted Sentiment Index (novel contribution)
- **Migration detection** — 3-tier rule-based pipeline (mention → signal → reason)
- **NetworkX** — directed migration graph with betweenness centrality

### Visualisation & Dashboard
- **Streamlit** — multipage dashboard
- **Plotly** — interactive charts (violin, heatmap, network graph)
- **Matplotlib / Seaborn** — notebook figures (saved at 300 DPI)

---

## 📈 Key Findings

> *Auto-generated from data — update after running all notebooks*

- **Most satisfied platform:** Identified from TWSI ranking
- **Highest competitor mention rate:** Reveals which platform users compare most
- **Top migration reason:** Pricing / Accuracy / Speed (from reason extraction)
- **Most discussed LDA topic:** Dominant discussion theme across all reviews
- **Platform gaining most users:** Net migration winner from directed graph

---

## 🔬 Novel Contributions

| # | Contribution | Description |
|---|---|---|
| NC1 | **TWSI** | Thumbs-Weighted Sentiment Index — community-endorsed satisfaction |
| NC2 | **5-Aspect ABSA** | Accuracy · Pricing · UI · Speed · Privacy radar (Notebook 03) |
| NC3 | **Migration graph** | Directed NetworkX graph of platform switching behaviour |
| NC4 | **3-Tier detection** | Mention → migration signal → reason categorisation pipeline |
| NC5 | **Platform scorecards** | 14-metric competitive intelligence per platform |

---

## 📂 Output Files

| File | Size | Contents |
|---|---|---|
| `reviews_clean.parquet` | ~60 MB | 22 new columns from preprocessing |
| `reviews_sentiment.parquet` | ~90 MB | 20 sentiment + ensemble columns |
| `reviews_topics.parquet` | ~95 MB | 5 LDA + BERTopic columns |
| `reviews_migration.parquet` | ~100 MB | 9 migration intelligence columns |
| `outputs/figures/*.png` | ~40 files | All notebook figures at 300 DPI |

---

## 🧪 CodeAlpha Task Coverage

| Task | Implementation |
|---|---|
| ✅ Web Scraping | Google Play scraper + Reddit PRAW (Notebook 05 Section 2) |
| ✅ EDA | Full 8-dimension audit across all 10 columns (Notebook 01) |
| ✅ Data Visualization | 40+ charts across all notebooks + 14-chart Streamlit dashboard |
| ✅ Sentiment Analysis | VADER + TextBlob + RoBERTa + TWSI ensemble (Notebook 03) |

---

## 📖 Dataset

**The Generative AI Ecosystem — 50K User Reviews (2026)**

- 50,000 app store reviews
- 5 platforms: ChatGPT · Gemini · Claude · MS Copilot · Perplexity
- 10,000 reviews per platform (perfectly balanced)
- 10 original columns → 66 total columns after full pipeline

---

## 👩‍💻 Project Info

**Type:** Final-year B.Tech CSE (Data Science) — CodeAlpha Internship Project  
**Dataset:** Generative AI Ecosystem 50K User Reviews 2026  
**Notebooks:** 5 · **Dashboard pages:** 5 · **Figures:** 41  

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
