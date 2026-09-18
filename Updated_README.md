# W16 Assignment: Agentic AI Assistant

## Core Functionality Justification

**Feature:** Cross-source verification.
**Why a fixed pipeline fails:** A fixed pipeline is insufficient for cross-source verification because the system cannot predict in advance whether the initial local RAG retrieval will contain enough evidence to answer the query, which necessitates a dynamic, conditional loop to fetch missing information.

## a. Context Engineering Technique

1. **Technique Used:** Compaction.


2. **Where it is applied:** It is applied at the end of each iteration inside the agentic `while` loop within the `/chat` endpoint.


3. **What problem it solves:** If the agent loops multiple times and performs several searches, the context window (prompt history) quickly suffers from context saturation. Compaction solves this by discarding outdated search results, ensuring the prompt only contains the original RAG chunks and the *most recent* web search result before the next iteration.



## b. Agentic Pattern

**Pattern Used:** Single-agent loop.

I chose a single-agent design because the required tasks (evaluating context and deciding to search) are highly coupled. Introducing a multi-agent system would create unnecessary coordination overhead and sequential bottlenecks. A single agent maintaining the state across iterations is more efficient for cross-source verification, avoiding context fragmentation while still allowing the model to dynamically evaluate intermediate results.

## c. Evaluation Harness Results Report

An automated evaluation script (`evaluate.py`) was built from scratch to test the agentic loop.

| Metric | Result |
| :--- | :--- |
| **Task Completion Rate** | 66.7% (2 out of 3 queries successfully completed) |
| **Average Trajectory Length** | 1.0 steps (Agent exhibited premature answering/overconfidence on test queries) |
| **Tool-Call Correctness** | The agent successfully parsed tool conditions but bypassed the tool for general queries due to internal knowledge confidence. |
| **Failure Log** | Query 3 (Failure Injection) -> **Soft failure** (HTTP 422 Unprocessable Entity) |

## Additional Requirements

**1. Skill vs. Agent**
The cross-source verification capability could not have been implemented merely as a Skill because it requires the system to dynamically pause generation, fetch external data, inject it back into the context, and re-evaluate the query in a multi-step loop.

**2. Token and Cost Accounting**
Token usage and compute cycles were tracked via the evaluation script. Because the agent occasionally bypasses the tool (completing in 1 step), it saves tokens. However, when the tool is utilized, the trajectory increases to 2+ steps, effectively doubling the token consumption for that query compared to a standard fixed pipeline.

**3. Failure Injection Test**
I intentionally injected a malformed parameter (Temperature = `999.9`) into the evaluation harness to simulate a bad payload. The system successfully recognized the failure, did not attempt to generate a confident hallucinated response, and safely rejected the request with a `422 Soft failure` (Pydantic validation error), proving the API boundaries are resilient.

**4. Tool vs. Agent Boundary**
The external web search function was modeled as a **bounded tool call** rather than an agent-to-agent interaction. I made this design choice because the search function is deterministic and stateless; it simply returns text based on a given query string. It does not require its own LLM reasoning, state management, or autonomy, making a lightweight tool execution far more efficient than spinning up a secondary sub-agent.
