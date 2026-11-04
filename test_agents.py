from modules.agents.workflow import run_farming_agent

# Test 1: RAG knowledge agent
result = run_farming_agent(
    "What are the recommended practices for wheat cultivation?"
)

print("\nAGENT:", result["agent"])
print("\nANSWER:\n", result["answer"])
print("\nSOURCES:\n", result["sources"])


# Test 2: Weather agent
weather_result = run_farming_agent(
    "Give me the current weather and irrigation information",
    district="Charkhi Dadri",
)

print("\nWEATHER AGENT:", weather_result["agent"])
print("\nWEATHER ANSWER:\n", weather_result["answer"])