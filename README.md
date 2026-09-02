# Wk15 AI Assignment: Containerized RAG Backend & Web UI

This repository contains the complete implementation for Task 1 and Task 2 of the AI assignment. It features a containerized FastAPI backend with a local Retrieval-Augmented Generation (RAG) pipeline using ChromaDB, paired with a modern web UI, all orchestrated seamlessly via Docker Compose.

---

## Project Structure

```text
Nirika_Wk14 assignment/
├── backend/
│   ├── data/
│   │   └── sample_doc.txt
│   ├── Dockerfile
│   ├── main.py
│   └── requirements.txt
├── frontend/
│   ├── Dockerfile
│   └── index.html
├── docker-compose.yml
└── .gitignore