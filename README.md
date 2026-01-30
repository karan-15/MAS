# MAS
Multi-Agent Survey Inference System

Automated survey opinion distribution prediction using retrieval-augmented, multi-stage inference over large-scale unstructured text.

Project Overview

This project implements a retrieval-augmented inference pipeline to predict survey answer distributions using evidence from a large document corpus.
The system combines lexical retrieval (BM25), dense retrieval (precomputed embeddings + FAISS), and LLM-based stance aggregation, evaluated using Jensen–Shannon divergence (JS) and Mean Average Precision (MAP).

The implementation follows the MAS (Multi-Agent Survey) competition specification and is designed for CPU-only execution.

Requirements

Python 3.9+ (tested up to 3.12)

CPU-only (no GPU required)

8GB RAM recommended

API key for LLM inference (Together AI or Hugging Face)

Python Version

Developed with: Python 3.9.18

Compatible with: Python 3.9–3.12

Recommended: Python 3.9.x for exact reproducibility

Installation
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

Data Setup

Note: Large data files are not included in the repository.

Required files:

data/documents.jsonl (~230K documents)

data/id_to_embedding.npz (precomputed dense embeddings)

data/dev.json (15 development questions)

data/test.json (10 test questions)

data/mini_dev.json (1-question sanity check)

These files are provided via course materials / Kaggle competition.

Index Building
python -m index.build config.yaml


This builds:

BM25 index (Whoosh)

Dense index (FAISS) from precomputed embeddings

Indexes are saved to:

index_artifacts/


Notes

Indexes are not committed to git

First-time build takes ~2–3 minutes on CPU

Running the System
Mini Sanity Check (Recommended First)
python -m mas_survey.run config_mini.yaml


1 question

BM25-only

Runtime: < 1 minute

Used to validate correctness and output format

Development Set
python -m mas_survey.run config.yaml


15 questions

Hybrid BM25 + dense retrieval

Runtime: ~25–30 minutes (CPU-only)

Output:

outputs/dev_predictions.csv

Evaluation (Development Set)
python -m mas_survey.eval.eval_js config.yaml


Achieved performance (verified):

Private JS Score: 0.187

MAP: 0.104

Test Set (Final Submission)
# Update config.yaml:
# questions_path: data/test.json
python -m mas_survey.run config.yaml


Output:

outputs/test_predictions.csv


This file is submitted to Kaggle.

Configuration Summary
config.yaml (Main)

Hybrid retrieval: BM25 + dense (FAISS)

RRF fusion

Top-100 evidence documents per question

LLM-based stance aggregation

config_mini.yaml (Debug)

BM25-only

Top-50 documents

No dense retrieval

Fast iteration and debugging

Output Format

Each row in the output CSV contains:

question: Survey question text

distribution: JSON object mapping options → probabilities

supports: JSON list of exactly 100 document IDs (50 in mini config)

This format strictly follows competition requirements.

Models Used
Retrieval

BM25: Whoosh

Dense embeddings: Precomputed (id_to_embedding.npz)

Vector index: FAISS (CPU)

Inference

LLM (stance classification):

Together AI (Qwen / Mistral variants), or

Hugging Face inference (small models for testing)

Ground-truth labels are not used during inference, only for evaluation.

System Architecture (Actual Implementation)

Implemented pipeline stages:

Query Expansion

Multiple semantic query variants per question

Hybrid Retrieval

BM25 + dense retrieval

Reciprocal Rank Fusion (RRF)

Candidate Filtering

Length filtering

Deduplication

Stance Inference

LLM-based document-level probability estimation

Aggregation

Probability normalization

Final distribution computation

Support document selection

Some agents shown in the assignment diagram (e.g. supervisor agent) were not enabled in the final run and are not claimed here.

Performance Characteristics

CPU-only Runtime

Mini config: ~45–60 seconds

Full dev set: ~25–30 minutes

Main bottleneck

LLM inference latency during stance classification

Reproducibility

Fixed random seed (seed = 42)

Deterministic retrieval and aggregation

Config-driven execution

No GPU dependencies

Project Structure
├── mas_survey/
│   ├── agents/              # Retrieval, stance, aggregation logic
│   ├── retrieval/           # BM25 + FAISS retrievers
│   ├── embeddings/          # Embedder interfaces
│   ├── llm/                 # LLM clients (Together / HF)
│   └── run.py               # Main pipeline entry point
├── index/
│   └── build.py             # Index builder
├── data/                    # Data files (not in git)
├── index_artifacts/         # Built indexes (not in git)
├── config.yaml
├── config_mini.yaml
├── requirements.txt
└── README.md

What This Project Demonstrates

Retrieval-Augmented Generation (RAG)

Hybrid lexical + dense retrieval

Large-scale document inference

Evidence-grounded prediction

Practical evaluation using JS divergence & MAP
