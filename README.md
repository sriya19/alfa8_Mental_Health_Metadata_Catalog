# alfa8_Mental_Health_Metadata_Catalog
DAEN 690 - 001 Group C - ALPHA8_Capstone project: Metadata Catalog for U.S. Mental Health Datasets
Perfect 👍 I reviewed your **`mental_health_qa_system.py`** file. It’s a Streamlit-based **Mental Health Q&A System** that integrates:

* File metadata analysis (via `file_metadata_analyzer.py`)
* A registry (SQLite) to store metadata and Q&A history
* LLM support (OpenAI + Anthropic Claude)
* Simulated MCP data sources
* Visualizations (Plotly dashboards)

Here’s a ready-to-use **README.md** for your repo:

---

# 🧠 Mental Health Q&A System

An interactive **Streamlit application** that enables healthcare researchers, policy analysts, and professionals to:

* Upload and analyze datasets
* Extract and store file-level metadata
* Query metadata and mental health data using **LLMs** (OpenAI or Claude)
* Visualize dataset quality, reliability, and Q&A confidence scores
* Track query history in a lightweight registry (SQLite)

---

## 🚀 Features

* **Metadata Extraction**: Analyze CSV, JSON, and TXT files for schema, quality, and structure.
* **Metadata Registry**: Store and retrieve file metadata, quality metrics, and reliability scores.
* **LLM-Powered Q&A**: Ask natural-language questions, answered by OpenAI GPT or Anthropic Claude with supporting context.
* **MCP Data Sources (Simulated)**: Includes PubMed, ClinicalTrials.gov, Mental Health Registries, and more.
* **Visual Dashboards**: Interactive charts for quality scores, file types, reliability, and Q&A confidence.
* **History Tracking**: Saves past Q&A sessions with timestamps, sources, and confidence levels.

---

## 🛠️ Installation

### 1. Clone this repository

```bash
git clone https://github.com/your-username/mental-health-qa-system.git
cd mental-health-qa-system
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate   # On Mac/Linux
# .venv\Scripts\activate    # On Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If you don’t have a `requirements.txt`, create one with:

```txt
streamlit
pandas
plotly
openai
anthropic
```

### 4. Add API keys (optional, for LLM Q&A)

Create a `.env` file or set environment variables:

```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
```

---

## ▶️ Usage

Run the Streamlit app:

```bash
streamlit run mental_health_qa_system.py
```

Access in your browser: [http://localhost:8501](http://localhost:8501)

---

## 📂 Project Structure

```
mental_health_qa_system.py      # Main Streamlit app
file_metadata_analyzer.py       # Metadata extraction logic
metadata_catalog.db             # SQLite registry (auto-created)
data/                           # (Optional) Folder for raw/processed datasets
```

---

## 📊 Application Tabs

1. **Upload & Analyze** – Upload datasets, extract metadata, and store in registry.
2. **Q&A Interface** – Ask mental health-related questions using LLMs.
3. **Dashboard** – View dataset quality scores, file types, reliability scores, and Q&A confidence.
4. **History** – Browse past Q&A sessions with timestamps and confidence metrics.

---

## 🔮 Roadmap

* Replace simulated MCP data sources with live APIs (PubMed, SAMHSA, CDC).
* Expand support for additional file types (Excel, SAS, SPSS).
* Deploy Streamlit app to Streamlit Cloud for external access.
* Integrate semantic search and ranking models for metadata discovery.

---

## 📜 License

This project is under the Apache 2.0 License unless otherwise specified.

---


