from typing import TypedDict, Any

from langgraph.graph import StateGraph, START, END

from config import GROQ_API_KEY, GROQ_MODEL
from modules.rag.qa_chain import answer_question

from modules.weather import get_weather_report
from modules.yield_predictor import (
    predict_wheat_yield,
    calculate_total_production,
)

from ml.predict import predict_image


# ============================================================
# WORKFLOW STATE
# ============================================================

class FarmingState(TypedDict):
    question: str
    category: str
    district: str
    image: Any
    yield_inputs: dict
    area_acres: float
    evidence: str
    answer: str
    sources: list
    agent: str


# ============================================================
# 1. QUESTION ROUTER
# ============================================================

def classify_question(state: FarmingState):

    question = state["question"].lower()

    if state.get("image") is not None:
        category = "crop_health"

    elif any(word in question for word in [
        "disease", "leaf", "yellow", "rust", "spots",
        "infection", "symptoms", "fungus"
    ]):
        category = "crop_health"

    elif any(word in question for word in [
        "weather", "rain", "temperature", "humidity",
        "irrigation", "forecast"
    ]):
        category = "weather"

    elif any(word in question for word in [
        "yield", "production", "harvest", "tonnes",
        "prediction", "output"
    ]):
        category = "yield"

    else:
        category = "farming_knowledge"

    return {
        "category": category,
        "agent": category,
    }


def route_agent(state: FarmingState):
    return state["category"]


# ============================================================
# 2. FARMING KNOWLEDGE AGENT - RAG
# ============================================================

def farming_knowledge_agent(state: FarmingState):

    result = answer_question(state["question"])

    sources = result.get("sources", [])

    return {
        "answer": result["answer"],
        "sources": sources,
        "evidence": result["answer"],
    }


# ============================================================
# 3. CROP HEALTH AGENT - RESNET18 + RAG
# ============================================================

def crop_health_agent(state: FarmingState):

    image = state.get("image")

    # If an image is provided, use the trained disease model.
    if image is not None:

        prediction = predict_image(image)

        evidence = (
            "Wheat disease model prediction:\n"
            f"{prediction}\n\n"
            "Treat this as a model prediction, not a confirmed "
            "laboratory diagnosis."
        )

        return {
            "evidence": evidence,
            "answer": str(prediction),
            "sources": [],
        }

    # If there is no image, use the existing agricultural RAG.
    result = answer_question(state["question"])

    return {
        "answer": result["answer"],
        "evidence": result["answer"],
        "sources": result.get("sources", []),
    }


# ============================================================
# 4. WEATHER AGENT - OPEN-METEO
# ============================================================

def weather_agent(state: FarmingState):

    district = state.get("district", "").strip()

    if not district:
        return {
            "evidence": (
                "A district is required to retrieve local weather. "
                "Ask the farmer to provide their Haryana district."
            ),
            "answer": "Please provide your district for a local weather report.",
            "sources": [],
        }

    report = get_weather_report(district)

    if not report:
        return {
            "evidence": "No weather report was returned.",
            "answer": "Weather data is currently unavailable for this district.",
            "sources": [],
        }

    current = report.get("current_weather", {})
    irrigation = report.get("irrigation", {})
    forecast = report.get("forecast")

    forecast_text = ""

    if forecast is not None:
        try:
            forecast_text = forecast.head(5).to_string(index=False)
        except Exception:
            forecast_text = str(forecast)

    evidence = f"""
District: {report.get('district', district)}

Current weather:
{current}

Irrigation information:
{irrigation}

Upcoming forecast:
{forecast_text}

Important: ET0 is reference evapotranspiration, not the actual
water requirement of the wheat crop.
"""

    return {
        "evidence": evidence,
        "answer": evidence,
        "sources": ["Open-Meteo weather data"],
    }


# ============================================================
# 5. YIELD AGENT - XGBOOST
# ============================================================

def yield_agent(state: FarmingState):

    input_data = state.get("yield_inputs") or {}

    required_fields = [
        "district",
        "year_start",
        "yield_lag_1",
        "yield_lag_2",
        "historical_avg_yield",
        "rainfall_mm",
        "temp_max_c",
        "temp_min_c",
        "humidity_pct",
    ]

    missing = [
        field for field in required_fields
        if field not in input_data or input_data[field] is None
    ]

    if missing:
        return {
            "evidence": f"Missing yield prediction inputs: {missing}",
            "answer": (
                "I need the following inputs for yield prediction: "
                + ", ".join(missing)
            ),
            "sources": [],
        }

    prediction = predict_wheat_yield(input_data)

    try:
        if isinstance(prediction, dict):
            yield_value = prediction.get(
                "yield_tonnes_per_hectare",
                prediction.get("prediction", prediction.get("yield"))
            )
        else:
            yield_value = prediction

        yield_value = float(yield_value)

        area_acres = state.get("area_acres") or 0

        production_text = ""

        if area_acres > 0:
            production = calculate_total_production(
                yield_value,
                area_acres,
            )

            production_text = (
                f"\nEstimated production for {area_acres} acres: "
                f"{production} tonnes"
            )

        evidence = (
            f"Predicted yield: {yield_value:.3f} tonnes/hectare"
            f"{production_text}\n\n"
            "This is an experimental district-level model estimate, "
            "not a guaranteed farm-level yield."
        )

    except (TypeError, ValueError):
        evidence = f"Yield model output: {prediction}"

    return {
        "evidence": evidence,
        "answer": evidence,
        "sources": ["KhetMitra XGBoost yield model"],
    }


# ============================================================
# 6. DECISION-MAKING AGENT
# ============================================================

def decision_maker_agent(state: FarmingState):

    from langchain_groq import ChatGroq

    evidence = state.get("evidence", "")
    question = state["question"]

    # If the evidence is empty, don't invent a recommendation.
    if not evidence.strip():
        return {
            "answer": "I could not retrieve enough information to provide an advisory.",
        }

    # For ordinary RAG answers, retain the original sourced answer.
    if state["category"] == "farming_knowledge":
        return {"answer": evidence}

    llm = ChatGroq(
        model=GROQ_MODEL,
        api_key=GROQ_API_KEY,
        temperature=0.2,
    )

    prompt = f"""
You are the final agricultural decision-making agent
for KhetMitra AI, a wheat farming advisory system for Haryana, India.

Farmer's question:
{question}

Agent findings:
{evidence}

Instructions:
1. Summarize the findings in simple language.
2. Only use the supplied findings.
3. Do not invent pesticide or fertilizer doses.
4. Do not claim a prediction is guaranteed.
5. Clearly mention uncertainty and missing information.
6. If the findings are insufficient, say what information is needed.
7. Give practical next steps where the evidence supports them.

Respond with:
- Findings
- Explanation
- Recommended next steps
- Important caution
"""

    response = llm.invoke(prompt)

    return {
        "answer": response.content,
    }


# ============================================================
# 7. BUILD LANGGRAPH WORKFLOW
# ============================================================

def build_farming_workflow():

    graph = StateGraph(FarmingState)

    graph.add_node("classifier", classify_question)
    graph.add_node("farming_knowledge", farming_knowledge_agent)
    graph.add_node("crop_health", crop_health_agent)
    graph.add_node("weather", weather_agent)
    graph.add_node("yield", yield_agent)
    graph.add_node("decision_maker", decision_maker_agent)

    graph.add_edge(START, "classifier")

    graph.add_conditional_edges(
        "classifier",
        route_agent,
        {
            "farming_knowledge": "farming_knowledge",
            "crop_health": "crop_health",
            "weather": "weather",
            "yield": "yield",
        },
    )

    graph.add_edge("farming_knowledge", "decision_maker")
    graph.add_edge("crop_health", "decision_maker")
    graph.add_edge("weather", "decision_maker")
    graph.add_edge("yield", "decision_maker")

    graph.add_edge("decision_maker", END)

    return graph.compile()


farming_workflow = build_farming_workflow()


# ============================================================
# 8. PUBLIC FUNCTION
# ============================================================

def run_farming_agent(
    question: str,
    district: str = "",
    image=None,
    yield_inputs=None,
    area_acres: float = 0,
):

    if not question or not question.strip():
        raise ValueError("Please enter a farming question.")

    initial_state = {
        "question": question.strip(),
        "category": "",
        "district": district,
        "image": image,
        "yield_inputs": yield_inputs or {},
        "area_acres": area_acres,
        "evidence": "",
        "answer": "",
        "sources": [],
        "agent": "",
    }

    result = farming_workflow.invoke(initial_state)

    return {
        "answer": result.get("answer", ""),
        "sources": result.get("sources", []),
        "agent": result.get("agent", ""),
    }