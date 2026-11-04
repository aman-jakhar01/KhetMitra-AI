
import streamlit as st
import joblib
import pandas as pd
import requests
from pathlib import Path

from config import (
    APP_NAME,
    APP_VERSION,
    APP_TAGLINE,
    DEFAULT_REGION,
    GROQ_MODEL,
    RAG_INDEX_DIR,
)

from ml.predict import predict_image
from modules.rag.qa_chain import answer_question, get_qa_info


# ==========================================
# RAG CONFIGURATION
# ==========================================

RAG_INDEX_PATH = Path(RAG_INDEX_DIR)


def rag_index_exists():
    """Check whether the prebuilt local FAISS index is available."""
    return (
        (RAG_INDEX_PATH / "index.faiss").exists()
        and (RAG_INDEX_PATH / "index.pkl").exists()
    )


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title=APP_NAME,
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================
# 2. CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.hero {
    background: linear-gradient(
        120deg,
        #164E32,
        #3B8050
    );
    padding: 35px;
    border-radius: 18px;
    color: white;
    margin-bottom: 30px;
}

.hero h1 {
    color: white;
    font-size: 40px;
}

.hero p {
    color: #E1F0E4;
    font-size: 17px;
}

div[data-testid="stMetric"] {
    background-color: white;
    border: 1px solid #E5EAE1;
    padding: 18px;
    border-radius: 12px;
}

.result-card {
    background-color: #F0F7F1;
    border: 1px solid #CDE4D1;
    padding: 22px;
    border-radius: 14px;
    margin-top: 15px;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# 3. SIDEBAR
# ==========================================

with st.sidebar:

    st.title("🌱 KhetMitra AI")

    st.caption("Your intelligent farming companion")

    st.divider()

    selected_page = st.radio(
        "Navigation",
        [
            "Overview",
            "Crop Doctor",
            "Yield Predictor",
            "Weather & Irrigation",
            "AI Farming Assistant",
            "AI Agents"
        ]
    )

    st.divider()

    st.markdown("### Farm Configuration")

    st.write(f"📍 {DEFAULT_REGION}")

    st.write("🌾 Wheat")

    st.divider()

    st.caption(f"Version {APP_VERSION}")


# ==========================================
# 4. OVERVIEW DASHBOARD
# ==========================================

if selected_page == "Overview":

    st.markdown("""
    <div class="hero">

        <h1>Welcome to KhetMitra AI 🌾</h1>

        <p>
        Smarter farming decisions powered by
        Machine Learning, Deep Learning and AI.
        </p>

    </div>
    """, unsafe_allow_html=True)

    st.subheader("Farm Intelligence Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Target Region", "Haryana")
    col2.metric("Supported Crop", "Wheat")
    col3.metric("Platform Version", APP_VERSION)

    st.divider()

    st.subheader("Our Intelligence Modules")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### 📷 Crop Doctor")

        st.write(
            "Identify potential wheat leaf conditions "
            "using Deep Learning and Computer Vision."
        )

        st.success("Status: Model integrated")

        st.markdown("### 📈 Yield Predictor")

        st.write(
            "Predict crop production using "
            "historical agricultural data."
        )

        st.success("Status: Model integrated")

        st.markdown("### 🌦️ Weather & Irrigation")

        st.write(
            "View live weather, a 7-day forecast, rainfall and reference "
            "evapotranspiration for wheat irrigation planning."
        )

        st.success("Status: Weather API integrated")

    with col2:

        st.markdown("### 🤖 AI Farming Assistant")

        st.write(
            "Ask farming questions and receive "
            "AI-powered agricultural guidance."
        )

        st.success("Status: RAG assistant integrated")

        st.markdown("### 🧠 Autonomous AI Agents")

        st.write(
            "Route farming questions to specialized LangGraph agents "
            "for crop health, weather, yield and document-grounded guidance."
        )

        st.success("Status: LangGraph workflow integrated")

    st.divider()

    st.caption(
        "KhetMitra AI | Smart Farming. Better Decisions."
    )


# ==========================================
# 5. CROP DOCTOR
# ==========================================

elif selected_page == "Crop Doctor":

    st.title("📷 Crop Doctor")

    st.write(
        "Upload a wheat leaf image to identify "
        "potential leaf conditions using our trained "
        "ResNet18 deep learning model."
    )

    st.divider()

    st.subheader("Upload Wheat Leaf Image")

    uploaded_file = st.file_uploader(
        "Choose a wheat leaf image",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear image of a single wheat leaf."
    )

    if uploaded_file is not None:

        col1, col2 = st.columns([1, 1])

        with col1:

            st.image(
                uploaded_file,
                caption="Uploaded Wheat Leaf",
                use_container_width=True
            )

        with col2:

            st.markdown("### Image Information")

            st.write(f"**Filename:** {uploaded_file.name}")
            st.write(f"**File type:** {uploaded_file.type}")

            st.write(
                "**Supported crop:** Wheat"
            )

            st.write(
                "**Model:** ResNet18 (Transfer Learning)"
            )

        st.divider()

        if st.button(
            "🔍 Analyze Wheat Leaf",
            type="primary",
            use_container_width=True
        ):

            try:

                with st.spinner(
                    "Analyzing image using the Deep Learning model..."
                ):

                    uploaded_file.seek(0)

                    result = predict_image(uploaded_file)

                st.success("Image analysis completed!")

                st.subheader("Analysis Result")

                result_col1, result_col2 = st.columns(2)

                with result_col1:
                    st.subheader("Predicted condition")
                    st.success(str(result.get("display_name", "Prediction returned")))

                with result_col2:
                    st.metric(
                        "Model confidence",
                        f'{result.get("confidence", "N/A")}%'
                    )

                st.divider()

                st.subheader("Class Probability Breakdown")

                probabilities = result.get("probabilities", {})
                if probabilities:
                    for condition, probability in probabilities.items():
                        st.write(f"**{condition}: {probability}%**")
                        st.progress(max(0, min(100, int(round(float(probability))))))
                else:
                    st.info("The model did not return class probabilities.")

                st.divider()

                st.warning(
                    """
                    **Important agricultural disclaimer**

                    This is an experimental AI model trained on
                    controlled wheat-leaf images. Its prediction
                    is not a confirmed diagnosis. In particular,
                    nitrogen-deficiency detection is less reliable
                    in our current evaluation.

                    Do not apply fertilizer or pesticides based
                    only on this prediction. Verify the condition
                    with an agricultural expert or soil test.
                    """
                )

            except Exception as error:

                st.error(
                    f"Prediction failed: {error}"
                )

    else:

        st.info(
            "Upload a wheat leaf image to begin analysis."
        )

    st.divider()

    st.caption(
        "Crop Doctor | Experimental wheat disease classification"
    )


# ==========================================
# 6. YIELD PREDICTOR
# ==========================================

elif selected_page == "Yield Predictor":

    st.title("📈 Wheat Yield Predictor")

    st.write(
        "Estimate district-level wheat yield using the trained "
        "Weather XGBoost model, then calculate production for your farm area."
    )

    MODEL_PATH = Path(__file__).resolve().parent / "models" / "wheat_yield_xgboost.joblib"

    @st.cache_resource
    def load_yield_model(model_path):
        return joblib.load(model_path)

    if not MODEL_PATH.exists():
        st.error(
            f"Trained model file not found: {MODEL_PATH}\n\n"
            "Place wheat_yield_xgboost.joblib inside the project's "
            "'models' folder."
        )
    else:
        try:
            artifact = load_yield_model(str(MODEL_PATH))
            yield_model = artifact["model"]
            model_features = artifact["features"]

            st.success(
                f"Model loaded: {artifact.get('model_name', 'Wheat Yield Model')} "
                f"| Training records: {artifact.get('training_rows', 'Not available')}"
            )

            st.caption(
                "Enter the exact district and historical/weather inputs required "
                "by the trained model. Do not guess missing values."
            )

            with st.form("wheat_yield_prediction_form"):
                st.subheader("1. District and season")

                district = st.text_input(
                    "District",
                    placeholder="Enter a Haryana district exactly as in the training data"
                )
                year_start = st.number_input(
                    "Season start year (e.g. 2022 for 2022–23)",
                    min_value=1900,
                    max_value=2100,
                    value=2022,
                    step=1
                )

                st.subheader("2. Historical yield information")
                lag1 = st.number_input(
                    "Previous season yield — lag 1 (tonnes/hectare)",
                    min_value=0.0, value=0.0, step=0.01, format="%.3f"
                )
                lag2 = st.number_input(
                    "Yield from two seasons earlier — lag 2 (tonnes/hectare)",
                    min_value=0.0, value=0.0, step=0.01, format="%.3f"
                )
                historical_avg = st.number_input(
                    "Historical average yield (tonnes/hectare)",
                    min_value=0.0, value=0.0, step=0.01, format="%.3f"
                )

                st.subheader("3. Wheat-season weather (October–March)")
                rainfall = st.number_input(
                    "Seasonal rainfall (mm)",
                    min_value=0.0, value=0.0, step=1.0, format="%.2f"
                )
                temp_max = st.number_input(
                    "Mean daily maximum temperature (°C)",
                    min_value=-20.0, max_value=60.0, value=0.0, step=0.1
                )
                temp_min = st.number_input(
                    "Mean daily minimum temperature (°C)",
                    min_value=-20.0, max_value=60.0, value=0.0, step=0.1
                )
                humidity = st.number_input(
                    "Mean relative humidity (%)",
                    min_value=0.0, max_value=100.0, value=0.0, step=0.1
                )

                st.subheader("4. Farm area")
                area_acres = st.number_input(
                    "Farm area (acres)",
                    min_value=0.01, value=1.0, step=0.5, format="%.2f"
                )

                submitted = st.form_submit_button(
                    "Predict Wheat Yield",
                    type="primary",
                    use_container_width=True
                )

            if submitted:
                if not district.strip():
                    st.error("Please enter a district.")
                elif any(value <= 0 for value in [lag1, lag2, historical_avg]):
                    st.error(
                        "Previous-season yields and the historical average must be "
                        "greater than zero. Zero rainfall and zero-degree temperatures "
                        "are valid inputs when they reflect the actual season."
                    )
                else:
                    input_values = {
                        "district": district.strip(),
                        "year_start": int(year_start),
                        "yield_lag_1": float(lag1),
                        "yield_lag_2": float(lag2),
                        "historical_avg_yield": float(historical_avg),
                        "rainfall_mm": float(rainfall),
                        "temp_max_c": float(temp_max),
                        "temp_min_c": float(temp_min),
                        "humidity_pct": float(humidity)
                    }

                    missing = [feature for feature in model_features if feature not in input_values]
                    if missing:
                        st.error(f"The saved model expects additional features: {missing}")
                    else:
                        try:
                            model_input = pd.DataFrame(
                                [{feature: input_values[feature] for feature in model_features}]
                            )
                            prediction = float(yield_model.predict(model_input)[0])

                            if prediction < 0:
                                st.warning(
                                    "The model returned a negative yield estimate. "
                                    "Please verify the inputs; production was not calculated."
                                )
                            else:
                                area_hectares = float(area_acres) * 0.404686
                                production_tonnes = prediction * area_hectares

                                st.divider()
                                st.subheader("Prediction results")

                                result_col1, result_col2, result_col3 = st.columns(3)
                                result_col1.metric(
                                    "Estimated yield",
                                    f"{prediction:.3f} tonnes/ha"
                                )
                                result_col2.metric(
                                    "Farm area",
                                    f"{area_hectares:.3f} hectares"
                                )
                                result_col3.metric(
                                    "Estimated production",
                                    f"{production_tonnes:.3f} tonnes"
                                )

                                st.metric(
                                    "Estimated production in kg",
                                    f"{production_tonnes * 1000:,.1f} kg"
                                )

                                st.warning(
                                    "This is an experimental district-level model, not a "
                                    "farm-specific forecast. It requires completed October–March "
                                    "seasonal weather and historical yield inputs. Do not use it "
                                    "as a pre-season prediction or as a guarantee of harvest."
                                )

                        except Exception as error:
                            st.error(f"Yield prediction failed: {error}")

        except Exception as error:
            st.error(f"Could not load the trained yield model: {error}")


# ==========================================
# 7. WEATHER & IRRIGATION
# ==========================================

elif selected_page == "Weather & Irrigation":

    st.title("🌦️ Weather & Irrigation")
    st.write(
        "Live weather and a 7-day forecast for Haryana wheat-growing areas, "
        "with a simple rainfall-versus-ET₀ irrigation planning indicator."
    )

    st.caption(
        "Weather data: Open-Meteo forecast models. District coordinates are "
        "geocoded; values represent a modelled location, not measurements from your field."
    )

    HARYANA_DISTRICTS = [
        "Ambala", "Bhiwani", "Charkhi Dadri", "Faridabad", "Fatehabad",
        "Gurugram", "Hisar", "Jhajjar", "Jind", "Kaithal", "Karnal",
        "Kurukshetra", "Mahendragarh", "Nuh", "Palwal", "Panchkula",
        "Panipat", "Rewari", "Rohtak", "Sirsa", "Sonipat", "Yamunanagar"
    ]

    WEATHER_CODES = {
        0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
        45: "Fog", 48: "Depositing rime fog", 51: "Light drizzle",
        53: "Moderate drizzle", 55: "Dense drizzle", 56: "Freezing drizzle",
        57: "Dense freezing drizzle", 61: "Slight rain", 63: "Moderate rain",
        65: "Heavy rain", 66: "Freezing rain", 67: "Heavy freezing rain",
        71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
        77: "Snow grains", 80: "Rain showers", 81: "Moderate rain showers",
        82: "Violent rain showers", 85: "Snow showers", 86: "Heavy snow showers",
        95: "Thunderstorm", 96: "Thunderstorm with hail", 99: "Thunderstorm with heavy hail"
    }

    @st.cache_data(ttl=1800, show_spinner=False)
    def get_haryana_coordinates(district_name):
        geocode_url = "https://geocoding-api.open-meteo.com/v1/search"
        response = requests.get(
            geocode_url,
            params={
                "name": district_name,
                "count": 20,
                "language": "en",
                "format": "json"
            },
            timeout=20
        )
        response.raise_for_status()
        results = response.json().get("results", [])

        for result in results:
            country_ok = result.get("country_code") == "IN"
            admin1 = (result.get("admin1") or "").strip().lower()
            if country_ok and admin1 == "haryana":
                return {
                    "latitude": result["latitude"],
                    "longitude": result["longitude"],
                    "place_name": result.get("name", district_name),
                    "admin1": result.get("admin1", "Haryana")
                }

        raise ValueError(
            f"Could not verify {district_name} as a location in Haryana. "
            "Please try another district."
        )

    @st.cache_data(ttl=1800, show_spinner=False)
    def get_weather_forecast(latitude, longitude):
        forecast_url = "https://api.open-meteo.com/v1/forecast"
        response = requests.get(
            forecast_url,
            params={
                "latitude": latitude,
                "longitude": longitude,
                "current": [
                    "temperature_2m", "relative_humidity_2m", "apparent_temperature",
                    "precipitation", "weather_code", "wind_speed_10m"
                ],
                "daily": [
                    "weather_code", "temperature_2m_max", "temperature_2m_min",
                    "precipitation_sum", "precipitation_probability_max",
                    "et0_fao_evapotranspiration"
                ],
                "timezone": "Asia/Kolkata",
                "forecast_days": 7,
                "temperature_unit": "celsius",
                "wind_speed_unit": "kmh",
                "precipitation_unit": "mm"
            },
            timeout=30
        )
        response.raise_for_status()
        payload = response.json()
        if "current" not in payload or "daily" not in payload:
            raise ValueError("Weather provider returned incomplete forecast data.")
        return payload

    selected_district = st.selectbox(
        "Select Haryana district",
        HARYANA_DISTRICTS,
        index=HARYANA_DISTRICTS.index("Charkhi Dadri")
    )

    fetch_weather = st.button("Refresh weather data", type="primary")
    if fetch_weather:
        # Clear only the two weather lookups; preserve unrelated cached data.
        get_haryana_coordinates.clear()
        get_weather_forecast.clear()

    try:
        with st.spinner(f"Fetching weather for {selected_district}..."):
            location = get_haryana_coordinates(selected_district)
            weather = get_weather_forecast(
                location["latitude"], location["longitude"]
            )

        current = weather["current"]
        daily = weather["daily"]

        st.subheader(f"Weather for {selected_district}, Haryana")
        st.caption(
            f"Model location: {location['place_name']} | "
            f"Coordinates: {location['latitude']:.3f}, {location['longitude']:.3f} | "
            f"Updated: {current.get('time', 'Not available')}"
        )

        current_code = current.get("weather_code")
        current_description = WEATHER_CODES.get(current_code, "Weather condition unavailable")

        metric1, metric2, metric3, metric4 = st.columns(4)
        metric1.metric(
            "Temperature", f"{current.get('temperature_2m', '—')} °C",
            help="Air temperature at approximately 2 metres above ground."
        )
        metric2.metric(
            "Feels like", f"{current.get('apparent_temperature', '—')} °C"
        )
        metric3.metric(
            "Humidity", f"{current.get('relative_humidity_2m', '—')} %"
        )
        metric4.metric(
            "Wind speed", f"{current.get('wind_speed_10m', '—')} km/h"
        )
        st.info(
            f"Current condition: **{current_description}** | "
            f"Recent precipitation: **{current.get('precipitation', '—')} mm**"
        )

        forecast_df = pd.DataFrame({
            "Date": daily["time"],
            "Condition": [WEATHER_CODES.get(code, "Unknown") for code in daily["weather_code"]],
            "Max temp (°C)": daily["temperature_2m_max"],
            "Min temp (°C)": daily["temperature_2m_min"],
            "Rainfall (mm)": daily["precipitation_sum"],
            "Rain chance (%)": daily["precipitation_probability_max"],
            "ET₀ (mm)": daily["et0_fao_evapotranspiration"]
        })

        st.divider()
        st.subheader("7-day forecast")
        st.dataframe(forecast_df, use_container_width=True, hide_index=True)

        st.subheader("Rainfall vs reference evapotranspiration")
        chart_df = forecast_df.set_index("Date")[["Rainfall (mm)", "ET₀ (mm)"]]
        st.bar_chart(chart_df, use_container_width=True)

        total_rain = float(pd.to_numeric(forecast_df["Rainfall (mm)"], errors="coerce").fillna(0).sum())
        total_et0 = float(pd.to_numeric(forecast_df["ET₀ (mm)"], errors="coerce").fillna(0).sum())
        rain_chance_max = float(pd.to_numeric(forecast_df["Rain chance (%)"], errors="coerce").fillna(0).max())
        water_balance = total_rain - total_et0

        st.divider()
        st.subheader("Irrigation planning indicator — next 7 days")
        summary1, summary2, summary3 = st.columns(3)
        summary1.metric("Forecast rainfall", f"{total_rain:.1f} mm")
        summary2.metric("Reference ET₀", f"{total_et0:.1f} mm")
        summary3.metric("Rainfall − ET₀", f"{water_balance:+.1f} mm")

        if total_rain >= total_et0 and total_rain > 0:
            st.success(
                "Forecast rainfall is at least as high as reference ET₀ over this period. "
                "Check actual field soil moisture and avoid irrigating automatically before or during rainfall."
            )
        elif total_rain < total_et0 and rain_chance_max < 30:
            st.warning(
                "Forecast rainfall is below reference ET₀ and the maximum daily rain probability is low. "
                "Inspect soil moisture and crop stage; irrigation may be needed if the root zone is drying."
            )
        else:
            st.info(
                "Forecast rainfall may partly offset atmospheric water demand. "
                "Review the daily forecast and check soil moisture before deciding when to irrigate."
            )

        st.caption(
            "ET₀ is reference evapotranspiration for a well-watered reference surface, not the exact "
            "water use of wheat. Actual irrigation depends on crop growth stage, crop coefficient, "
            "effective rainfall, soil type, root-zone moisture, irrigation method and recent watering. "
            "This screen does not calculate an irrigation depth or replace field inspection."
        )

    except requests.RequestException as error:
        st.error(
            "Could not retrieve weather data. Check your internet connection and try again. "
            f"Technical detail: {error}"
        )
    except Exception as error:
        st.error(f"Weather module error: {error}")


# ==========================================
# 8. AI FARMING ASSISTANT - QWEN RAG CHAT
# ==========================================

elif selected_page == "AI Farming Assistant":

    st.title("🤖 KhetMitra AI — Wheat Farming Assistant")

    st.write(
        "Ask questions about wheat cultivation. KhetMitra searches the "
        "preloaded ICAR and Government PDF knowledge base and uses Qwen "
        "through Groq to generate an evidence-grounded answer."
    )

    st.info(
        "The knowledge documents are preloaded by the project developer. "
        "Farmers do not need to upload PDFs. Answers are limited to the "
        "content retrieved from the current knowledge base."
    )

    st.caption(
        f"LLM: {GROQ_MODEL} | "
        "Embeddings: all-MiniLM-L6-v2 | "
        "Vector database: FAISS"
    )

    index_exists = rag_index_exists()

    if index_exists:
        st.success("Wheat knowledge base is ready.")
    else:
        st.error(
            "The FAISS knowledge index was not found. "
            "Run `uv run python build_knowledge_base.py` from D:\\agri "
            "to build it before using the assistant."
        )

    try:
        qa_info = get_qa_info()
        with st.expander("Knowledge assistant configuration"):
            st.write(f"**Provider:** {qa_info['provider']}")
            st.write(f"**Model:** {qa_info['model']}")
            st.write(f"**Retrieval:** {qa_info['retrieval_method']}")
            st.write(f"**Retrieved passages per question:** {qa_info['retrieval_k']}")
            st.write(f"**Index location:** `{RAG_INDEX_PATH}`")
    except Exception as error:
        st.warning(f"Could not read assistant configuration: {error}")

    st.divider()

    if "rag_chat_history" not in st.session_state:
        st.session_state["rag_chat_history"] = []

    chat_controls_col, _ = st.columns([1, 5])

    with chat_controls_col:
        if st.button(
            "🗑️ Clear conversation",
            use_container_width=True,
            disabled=not st.session_state["rag_chat_history"]
        ):
            st.session_state["rag_chat_history"] = []
            st.rerun()

    # Render previous conversation turns.
    for turn in st.session_state["rag_chat_history"]:
        with st.chat_message("user"):
            st.markdown(turn["question"])

        with st.chat_message("assistant"):
            st.markdown(turn["answer"])

            if turn.get("sources"):
                with st.expander("📚 Sources used"):
                    for source in turn["sources"]:
                        st.markdown(
                            f"- **{source['document']}** — "
                            f"page {source['page']} "
                            f"({source['category']})"
                        )

            st.caption(f"Model: {turn.get('model', GROQ_MODEL)}")

    if not index_exists:
        st.chat_input(
            "Ask a wheat farming question...",
            disabled=True
        )
    else:
        user_question = st.chat_input(
            "Ask a wheat farming question..."
        )

        if user_question:
            with st.chat_message("user"):
                st.markdown(user_question)

            with st.chat_message("assistant"):
                with st.spinner(
                    "Searching the farming PDFs and asking Qwen..."
                ):
                    try:
                        result = answer_question(user_question)

                        answer_text = result.get(
                            "answer",
                            "The assistant did not return an answer."
                        )
                        sources = result.get("sources", [])
                        model_name = result.get("model", GROQ_MODEL)

                        st.markdown(answer_text)

                        if sources:
                            with st.expander("📚 Sources used"):
                                for source in sources:
                                    st.markdown(
                                        f"- **{source['document']}** — "
                                        f"page {source['page']} "
                                        f"({source['category']})"
                                    )
                        else:
                            st.warning(
                                "No source passages were returned for this answer."
                            )

                        st.caption(f"Model: {model_name}")

                        st.session_state["rag_chat_history"].append({
                            "question": user_question,
                            "answer": answer_text,
                            "sources": sources,
                            "model": model_name
                        })

                    except Exception as error:
                        st.error(
                            f"Could not generate an answer: {error}"
                        )

    st.divider()

    st.caption(
        "Agricultural safety: AI-generated information is not a substitute "
        "for local agricultural expert advice. Verify pesticide and fertilizer "
        "recommendations against current official guidance and product labels. "
        "If the available PDFs do not support an answer, KhetMitra should say so."
    )


# ==========================================
# 9. AI AGENTS - LANGGRAPH WORKFLOW
# ==========================================

elif selected_page == "AI Agents":

    st.title("🧠 KhetMitra Autonomous Farming Agents")
    st.write(
        "Choose a farming task. LangGraph routes it to the appropriate "
        "agent and returns an evidence-based response."
    )

    st.caption(
        "Available integrations: FAISS + Qwen RAG, ResNet18 crop classifier, "
        "Open-Meteo weather and the experimental XGBoost yield model."
    )

    st.info(
        "The agents provide decision support, not confirmed diagnoses or "
        "guaranteed yield estimates. Check field conditions and official "
        "agricultural guidance before acting."
    )

    task = st.selectbox(
        "Choose an agent task",
        [
            "Farming knowledge question",
            "Weather & irrigation analysis",
            "Wheat leaf health analysis",
            "Wheat yield prediction",
        ],
        key="agent_task_selector",
    )

    HARYANA_DISTRICTS = [
        "Ambala", "Bhiwani", "Charkhi Dadri", "Faridabad", "Fatehabad",
        "Gurugram", "Hisar", "Jhajjar", "Jind", "Kaithal", "Karnal",
        "Kurukshetra", "Mahendragarh", "Nuh", "Palwal", "Panchkula",
        "Panipat", "Rewari", "Rohtak", "Sirsa", "Sonipat", "Yamunanagar"
    ]

    with st.form("autonomous_agent_form"):
        district = ""
        uploaded_leaf = None
        yield_inputs = None
        area_acres = 0.0

        if task == "Farming knowledge question":
            agent_question = st.text_area(
                "Your wheat farming question",
                placeholder="Example: Which wheat cultivation practices does ICAR recommend?",
                height=110,
            )

        elif task == "Weather & irrigation analysis":
            district = st.selectbox(
                "Haryana district",
                HARYANA_DISTRICTS,
                index=HARYANA_DISTRICTS.index("Charkhi Dadri"),
                key="agent_weather_district",
            )
            agent_question = st.text_area(
                "What would you like to know?",
                value="Analyse the current weather and explain what I should check before deciding whether to irrigate.",
                height=100,
            )

        elif task == "Wheat leaf health analysis":
            uploaded_leaf = st.file_uploader(
                "Upload a clear wheat leaf image",
                type=["jpg", "jpeg", "png"],
                key="agent_leaf_upload",
            )
            agent_question = st.text_area(
                "Describe what you observe (optional)",
                value="Analyse this wheat leaf image and explain the model's prediction.",
                height=90,
            )

        else:
            st.caption(
                "Enter the same historical and seasonal weather features used "
                "by the trained model. Use actual values, not guesses."
            )
            district = st.text_input(
                "District (match the training data)",
                placeholder="Example: Hisar",
            )
            year_start = st.number_input(
                "Season start year",
                min_value=2000,
                max_value=2100,
                value=2025,
                step=1,
            )
            lag1 = st.number_input(
                "Previous season yield (tonnes/hectare)",
                min_value=0.0,
                value=0.0,
                step=0.01,
            )
            lag2 = st.number_input(
                "Yield two seasons earlier (tonnes/hectare)",
                min_value=0.0,
                value=0.0,
                step=0.01,
            )
            historical_avg = st.number_input(
                "Historical average yield (tonnes/hectare)",
                min_value=0.0,
                value=0.0,
                step=0.01,
            )
            rainfall = st.number_input(
                "Seasonal rainfall (mm)",
                min_value=0.0,
                value=0.0,
                step=1.0,
            )
            temp_max = st.number_input(
                "Mean daily maximum temperature (°C)",
                min_value=-20.0,
                max_value=60.0,
                value=0.0,
                step=0.1,
            )
            temp_min = st.number_input(
                "Mean daily minimum temperature (°C)",
                min_value=-20.0,
                max_value=60.0,
                value=0.0,
                step=0.1,
            )
            humidity = st.number_input(
                "Mean relative humidity (%)",
                min_value=0.0,
                max_value=100.0,
                value=0.0,
                step=0.1,
            )
            area_acres = st.number_input(
                "Farm area (acres)",
                min_value=0.01,
                value=1.0,
                step=0.5,
            )
            agent_question = (
                "Estimate wheat yield using the supplied district, "
                "historical yield and seasonal weather inputs."
            )
            yield_inputs = {
                "district": district.strip(),
                "year_start": int(year_start),
                "yield_lag_1": float(lag1),
                "yield_lag_2": float(lag2),
                "historical_avg_yield": float(historical_avg),
                "rainfall_mm": float(rainfall),
                "temp_max_c": float(temp_max),
                "temp_min_c": float(temp_min),
                "humidity_pct": float(humidity),
            }

        run_agents = st.form_submit_button(
            "Run farming agent",
            type="primary",
            use_container_width=True,
        )

    if run_agents:
        if task == "Wheat leaf health analysis" and uploaded_leaf is None:
            st.error("Please upload a wheat leaf image.")
        elif task == "Wheat yield prediction" and not district.strip():
            st.error("Please enter a district.")
        elif task == "Wheat yield prediction" and any(
            value <= 0 for value in [lag1, lag2, historical_avg]
        ):
            st.error(
                "Previous-season yields and historical average must be greater "
                "than zero. Zero rainfall and zero-degree temperatures can be valid."
            )
        elif not agent_question.strip():
            st.error("Please enter a question.")
        else:
            try:
                # Import lazily so other app pages remain usable if agent dependencies fail.
                from modules.agents.workflow import run_farming_agent

                with st.spinner("LangGraph is routing the task and coordinating the agent..."):
                    result = run_farming_agent(
                        question=agent_question,
                        district=district,
                        image=uploaded_leaf,
                        yield_inputs=yield_inputs,
                        area_acres=area_acres,
                    )

                st.session_state["latest_agent_result"] = {
                    "task": task,
                    "question": agent_question,
                    "result": result,
                }

            except Exception as error:
                st.error(f"Agent workflow failed: {error}")
                st.exception(error)

    latest = st.session_state.get("latest_agent_result")

    if latest:
        st.divider()
        st.subheader("Agent response")

        result = latest["result"]
        agent_name = result.get("agent", "farming_agent")
        st.caption(f"Selected task: {latest['task']} | Routed agent: {agent_name}")

        st.markdown(result.get("answer", "The agent did not return an answer."))

        sources = result.get("sources", [])
        if sources:
            with st.expander("📚 Evidence and sources", expanded=True):
                for source in sources:
                    if isinstance(source, dict):
                        st.markdown(
                            f"- **{source.get('document', 'Source')}** — "
                            f"page {source.get('page', 'N/A')} "
                            f"({source.get('category', 'uncategorized')})"
                        )
                    else:
                        st.markdown(f"- {source}")

        if latest["task"] == "Wheat leaf health analysis":
            st.warning(
                "The ResNet18 output is an experimental image-classification "
                "prediction, not a confirmed diagnosis. Do not apply chemicals "
                "or fertilizer based only on this result."
            )
        elif latest["task"] == "Wheat yield prediction":
            st.warning(
                "This is an experimental district-level model estimate, not a "
                "farm-specific forecast or a guarantee of harvest."
            )
        elif latest["task"] == "Weather & irrigation analysis":
            st.warning(
                "ET₀ is reference evapotranspiration, not the exact water use "
                "of wheat. Check crop stage and root-zone soil moisture before "
                "making an irrigation decision."
            )

    st.divider()
    st.caption(
        "KhetMitra AI Agents | LangGraph orchestration with ML, weather and RAG tools"
    )
