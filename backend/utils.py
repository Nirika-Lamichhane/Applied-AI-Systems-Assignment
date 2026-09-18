def mock_web_search(query: str) -> str:
    """
    Searches the web for up-to-date information. 
    Use this tool ONLY if the local ChromaDB documents do not contain enough evidence to answer the user's prompt.
    """
    print(f"🕵️ Agent decided to search the web for: {query}")
    
    # Simulating a web search result for testing purposes
    if "W16" in query or "agent" in query:
        return "Web search result: Task 3 requires implementing an agentic loop with cross-source verification."
    else:
        return f"Web search result: Found additional external context regarding {query}."