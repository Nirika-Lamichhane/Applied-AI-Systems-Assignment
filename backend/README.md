## MLOps Track B Documentation

### a. Environment & Reproducibility (uv)
We migrated the project dependency management to `uv`. This solved the issue of resolving deep, conflicting dependencies between FastAPI, MLflow, and Pydantic that frequently occur in standard `pip` environments. It locks dependencies via `uv.lock`, ensuring that cloning this repository and running `uv sync` will perfectly replicate the working environment on any machine.

### b. Experiment Tracking Strategy (MLflow)
We tracked custom agentic loops by logging parameters (`prompt_version`, `model`, `temperature`), metrics (`iterations`, `latency`), and deep JSON trace artifacts (`agent_trace.json`) to a local SQLite MLflow database. 
- **v1**: Caught a silent `404` API error where the agent fell back on an exception block.
- **v2**: Updated the model string to `gemini-2.5-flash` and introduced a security rule for the CatalogLens password. It successfully blocked the password but took 2 iterations and a web search.
- **v3**: Optimized the prompt logic to synthesize answers faster. It achieved single-step resolution (1 iteration) without unnecessary tool calls, proving `v3` is the most efficient configuration.

### c. Monitoring & Drift Strategy (Evidently AI)
We established a regression testing pipeline using a "Golden Dataset" of approved questions and answers. The "reference" represents the perfect expected text, and the "current" data represents the live API responses from our `v3` agent. 
We monitored `TestColumnDrift` and `TestNumberOfMissingValues` via the Evidently Test Suite. The resulting pass/fail percentage is automatically logged back into MLflow as the `pct_tests_passed` metric, and the `evidently_report.html` artifact is attached to the run for visual inspection. If drift crosses the failure threshold in a CI/CD pipeline, the new prompt version would be rejected.