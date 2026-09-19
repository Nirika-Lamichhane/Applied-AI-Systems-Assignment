# W17 Track B: Agentic AI MLOps Pipeline

This repository contains the complete implementation for **Track B: Agentic AI MLOps**. It extends a W15/W16 FastAPI-based generative AI assistant (powered by Gemini) with production-grade MLOps disciplines: environment reproducibility, experiment tracking, deep trace observability, and automated regression testing.

## 🚀 Project Overview

The system features an autonomous AI agent capable of tool-use (e.g., local RAG and mock web search). This week's integration wraps the agent in a robust MLOps pipeline using three primary tools:
1. **`uv`**: For hermetic environment management and dependency locking.
2. **`MLflow`**: For logging agent configurations, prompt iterations, and deep JSON execution traces.
3. **`Evidently AI`**: For automated regression testing and structural drift monitoring against a golden dataset.

---

## ⚙️ Setup & Execution Instructions

This project relies on `uv` to guarantee exact dependency replication. 

**1. Clone the specific Track B branch and Install Dependencies**
`git clone -b track-b-mlops <your-repo-link>`
`cd backend`
`uv sync`

**2. Start the FastAPI Agentic Backend**
Ensure your `GEMINI_API_KEY` is set in your environment, then start the server:
`uv run uvicorn main:app --reload --port 8000`

**3. Run the MLflow Evaluation Harness**
Execute the test queries to trigger the agentic loops, generate `agent_trace.json` files, and log runs to the local SQLite database:
`uv run python evaluate.py`
*To view the dashboard, run `uv run python -m mlflow ui --backend-store-uri sqlite:///mlflow.db` and visit `http://127.0.0.1:5000`.*

**4. Run the Evidently AI Regression Suite**
Execute the automated QA script to compare live API responses against the golden dataset:
`uv run python regression_test.py`
*To view the results, open the generated `evidently_report.html` in any web browser.*

---

## 📋 MLOps Documentation (Assignment Rubric)

### a. Environment & Reproducibility (uv)

**The Problem Solved:** 
Building generative AI backends involves complex, highly volatile dependency chains—especially when combining frameworks like `FastAPI`, `MLflow`, `Pydantic`, and LLM SDKs (where v1 vs v2 conflicts frequently break environments). Standard `pip` setups are prone to silent breakages when transitive dependencies update remotely. By migrating to `uv`, this project eliminates dependency resolution bottlenecks.

**Reproducibility:**
The environment is strictly hermetic. The exact state of the working environment is captured in the committed `pyproject.toml` and `uv.lock` files. Running `uv sync` from a clean clone guarantees one-command reproducibility on any machine, installing the exact same package hashes and preventing the classic "it works on my machine" failure state.

### b. Experiment Tracking Strategy (MLflow)

**Variables and Metrics:**
Because this is an Agentic AI system, there is no "trained model." Instead, our primary experimental variable was the **system instruction (prompt version)** and the model configuration. 
For each experiment, we tracked:
*   **Parameters:** `prompt_version`, `model`, `temperature`.
*   **Metrics:** `iterations` (number of loops) and task success rate.
*   **Artifacts:** `agent_trace.json` (a detailed, step-by-step record of tool calls, arguments, reasoning, and raw results).

**Version Comparison & Trace Diagnosis:**

* **v1 (`gemini-1.5-flash`): Failure.** The trace artifact revealed a silent `404 API Error` because the model string was deprecated. The code's fallback block caught it, but the trace proved the AI agent loop never successfully started.
* **v2 (`gemini-2.5-flash` + Security Rule): Partial Success.** Added a strict instruction to block password extraction (`"ANSWER: Access Denied"`). The trace showed this succeeded in 1 step. However, for general queries, the trace revealed the agent lacked confidence in the local RAG context, taking **2 iterations** and a web search tool call, eventually hitting a Google API rate limit.
* **v3 (`gemini-2.5-flash` + Logic Optimization): Optimal.** We optimized the prompt to explicitly instruct the agent to synthesize concise answers from local context *first*, and only search if facts were missing. The trace proved the agent completed complex queries in exactly **1 iteration**.

**Conclusion:** 
**Version 3 is the winning configuration.** By tracking the deep execution traces, we diagnosed tool-use inefficiency in v2. The v3 prompt optimized the loop, trading speculative exploratory depth for lower latency, fewer token costs, and API rate-limit avoidance.

### c. Monitoring & Regression Testing (Evidently AI)

**Reference vs. Current Setup:**
To monitor the agent's behavior in production, we established a regression testing pipeline to detect degradation in logic or output structure.
*   **Reference Data ("Golden Dataset"):** A fixed Pandas DataFrame containing representative queries paired with approved, perfect responses (e.g., testing the strict security rules).
*   **Current Data (Production):** Live responses fetched dynamically by executing the same queries against the running `v3` FastAPI backend.

**Monitored Metrics & Evaluation:**
We used the `Evidently Test Suite` to evaluate the data:
1.  **Structural Integrity:** Tests like `TestNumberOfMissingValues` and `TestNumberOfEmptyRows` ensure the LLM didn't timeout, crash, or return empty JSON payloads (a common failure mode in agentic loops).
2.  **Regression Drift:** The suite runs a reference-based comparison (Reference vs. Current) to ensure the live agent's output maintains structural and factual alignment with the golden dataset.

**Report & Action on Drift:**
The evaluation outputs an `evidently_report.html` file (saved as an MLflow artifact) and logs a `pct_tests_passed` metric. In our final run, the system achieved a **100% Pass Rate**. 
**Action:** In a production CI/CD pipeline, if drift crosses a concerning threshold (e.g., the test suite fails because the agent starts hallucinating formats or leaking passwords), the pipeline would immediately block the new prompt version from entering the MLflow registry and alert the MLOps engineer to review the HTML report.