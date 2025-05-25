import streamlit as st
from ir_module.retriever import IRRetriever
from ai_module.a_star import graph, coordinates, disease_specialty, find_nearest_hospital
from streamlit_folium import st_folium
import folium
from ml_module.predictor import predict, predict_diabetes
from ai_module.a_star import hospital_specialists
from ai_module.a_star import get_coord


# -------------------
# Page Config
# -------------------
st.set_page_config(
    page_title="Medi-Assist",
    layout="centered",
    page_icon="🩺"  
)

# -------------------
# Custom CSS Styling
# -------------------
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    header {
        visibility: hidden;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .chat-box {
        border-radius: 15px;
        padding: 1.5rem;
        background-color: white;
        color: #222222;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
div.stButton > button {
    background-color: #007bff !important;
    color: white !important;
    padding: 0.6em 1.2em !important;
    border-radius: 8px !important;
    border: none !important;
    font-weight: bold !important;
    transition: background-color 0.3s ease !important;
    outline: none !important;
    box-shadow: none !important;
    text-shadow: none !important;
}

div.stButton > button:hover {
    background-color: #0056b3 !important; /* Darker blue on hover */
    color: white !important;              /* Keep text white */
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2) !important; /* Subtle hover shadow */
    transform: translateY(-2px) !important;  /* Slight lift on hover */
    transition: all 0.2s ease-in-out !important;
}


div.stButton > button:hover,
div.stButton > button:focus,
div.stButton > button:active,
div.stButton > button:visited {
    background-color: #0056b3 !important;
    color: white !important;
    outline: none !important;
    box-shadow: none !important;
    text-shadow: none !important;
}

    
    </style>
""", unsafe_allow_html=True)

# -------------------
# Title
# -------------------
st.markdown("<h1 style='text-align: center; color: white;'>🩺 Medi-Assist</h1>", unsafe_allow_html=True)
st.markdown("### Your Intelligent Medical Assistant\n", unsafe_allow_html=True)

# -------------------
# Sidebar Navigation
# -------------------
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Go to", ["Ask Medi-Assist", "Disease Predictor", "Find Nearby Doctor"])

# -------------------
# Ask Medi-Assist (IR + RAG)
# -------------------
if page == "Ask Medi-Assist":
    st.markdown("#### 🤖 Ask any medical question")

    if "retriever" not in st.session_state:
        ir = IRRetriever()
        ir.load_documents()
        ir.embed_and_index()
        st.session_state.retriever = ir

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.chat_input("Ask Medi-Assist anything medical...")

    if user_input:  
        with st.spinner("Thinking..."):
            results = st.session_state.retriever.search(user_input, top_k=3, threshold=0.2)

            if not results:
                retrieved_info = "*No relevant documents found.*"
                combined_context = ""
            else:
                retrieved_info = "##### 🔍 Top Retrieved Documents & Scores:\n"
                for name, _, score in results:
                    retrieved_info += f"- `{name}` (Score: {score:.3f})\n"
                combined_context = "\n".join([text for _, text, _ in results])

            answer = st.session_state.retriever.generate_answer(user_input, combined_context)

            st.session_state.chat_history.append({
                "user": user_input,
                "assistant": answer,
                "retrieved_info": retrieved_info
            })

    for message in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(message["user"])
        with st.chat_message("assistant"):
            if message["retrieved_info"]:
                st.markdown(message["retrieved_info"])
            st.markdown("#### 💬 Response:")
            st.markdown(f"<div class='chat-box'>{message['assistant']}</div>", unsafe_allow_html=True)


# ------------------- Find Nearby Doctor (A* Pathfinding) -------------------
elif page == "Find Nearby Doctor":
    st.markdown("#### 🧭 Find Nearby Doctor using A* Pathfinding")

    if "result" not in st.session_state:
        st.session_state.result = None
    if "path" not in st.session_state:
        st.session_state.path = None
    if "total_distance" not in st.session_state:
        st.session_state.total_distance = None

    st.markdown("Select a disease and your current location to find the nearest hospital with a relevant specialist using the A* algorithm.")

    selected_disease = st.selectbox("Select your disease:", sorted(disease_specialty.keys()))

    hospital_names = set(hospital_specialists.keys())
    city_locations = sorted([loc for loc in coordinates.keys() if loc not in hospital_names])

    start_location = st.selectbox("Select your starting location:", city_locations)

    if st.button("Find Nearest Hospital"):
        required_specialist = disease_specialty[selected_disease]
        path, total_distance, specialist_found = find_nearest_hospital(start_location, required_specialist)

        if path:
            if specialist_found:
                st.session_state.result = f"✅ Found a {required_specialist} at *{path[-1]}*!"
            else:
                st.session_state.result = f"⚠️ No hospital with a {required_specialist} was reachable. Showing nearest available hospital at *{path[-1]}* instead."

            st.session_state.path = path
            st.session_state.total_distance = total_distance

            if len(path) == 2 and (path[0], path[1]) not in [(a, b) for a in graph for b in graph[a]]:
                st.session_state.result += " (using direct distance fallback)"
        else:
            st.session_state.result = "❌ No hospital found from your starting location."
            st.session_state.path = None
            st.session_state.total_distance = None

    if st.session_state.result:
        if st.session_state.path:
            st.success(st.session_state.result)
            st.markdown("### 🗺️ Route:")
            for i in range(len(st.session_state.path) - 1):
                road = graph.get(st.session_state.path[i], {}).get(st.session_state.path[i+1], {}).get("road", "Unknown Road")
                st.write(f"{i+1}. {st.session_state.path[i]} → {st.session_state.path[i+1]} via {road}")
            st.write(f"{len(st.session_state.path)}. Destination: *{st.session_state.path[-1]}*")

            st.markdown(f"📏 *Total distance:* {st.session_state.total_distance} km")

            st.markdown("### 🗺️ Map View:")
            center_coords = coordinates.get(start_location, (33.6007, 73.0679))
            m = folium.Map(location=center_coords, zoom_start=12, tiles="CartoDB Positron")

            points = []
            for idx, loc in enumerate(st.session_state.path):
                coord = get_coord(loc)
                if coord:
                    points.append(coord)
                    if idx == 0:
                        color = 'green'
                        popup = f"Start: {loc}"
                    elif idx == len(st.session_state.path) - 1:
                        color = 'red'
                        popup = f"Hospital: {loc}"
                    else:
                        color = 'blue'
                        popup = loc
                    folium.Marker(
                        location=coord,
                        popup=popup,
                        icon=folium.Icon(color=color)
                    ).add_to(m)

            if len(points) >= 2:
                folium.PolyLine(points, color='blue', weight=5, opacity=0.7).add_to(m)

            st_folium(m, width=700, height=500)

        else:
            st.error(st.session_state.result)
            
            
            
# DISEASE PREDICTOR
elif page == "Disease Predictor":
    st.subheader("🎯 Disease Predictor")

    from ml_module.predictor import predict, predict_diabetes

    cancer_type = st.selectbox("Choose Disease Type", ["Diabetes", "Prostate", "Lung", "Skin"])

    if cancer_type == "Diabetes":
        st.subheader("🧪 Diabetes Prediction")
        st.markdown("#### Input Patient Details")
        pregnancies = st.number_input("Pregnancies (avg: 3)", min_value=0, value=3)
        glucose = st.number_input("Glucose (avg: 120 mg/dL)", min_value=0.0, value=120.0)
        bp = st.number_input("Blood Pressure (avg: 70 mmHg)", min_value=0.0, value=70.0)
        skin_thick = st.number_input("Skin Thickness (avg: 20 mm)", min_value=0.0, value=20.0)
        insulin = st.number_input("Insulin (avg: 80 µU/mL)", min_value=0.0, value=80.0)
        bmi = st.number_input("BMI (avg: 32.0)", min_value=0.0, value=32.0)
        dpf = st.number_input("Diabetes Pedigree Function (avg: 0.47)", min_value=0.0, value=0.47)
        age = st.number_input("Age (avg: 33)", min_value=1, max_value=120, value=33)

        if st.button("Predict Diabetes"):
            features = [pregnancies, glucose, bp, skin_thick, insulin, bmi, dpf, age]
            probability = predict_diabetes(features)
            st.write(f"Diabetes Probability: {probability:.2%}")
            if probability >= 0.5:
                st.error("🟥 High Risk of Diabetes")
            else:
                st.success("✅ Low Risk of Diabetes")

    elif cancer_type == "Prostate":
        st.subheader("🧬 Prostate Cancer Prediction")
        st.markdown("#### Input Patient Details")
        age = st.number_input("Age (avg: 65)", min_value=0, max_value=120, value=65)
        psa = st.number_input("PSA Level (avg: 1.5 ng/mL)", value=1.5)
        bmi = st.number_input("BMI (avg: 26.5)", value=26.5)
        vol = st.number_input("Prostate Volume (avg: 35 cm³)", value=35.0)

        if st.button("Predict Prostate Cancer"):
            result, probability = predict("prostate", [age, psa, bmi, vol])
            st.write(f"Cancer Probability: {probability:.2%}")
            if result:
                st.error("🟥 Cancer Detected")
            else:
                st.success("✅ No Cancer Detected")

    elif cancer_type == "Lung":
        st.subheader("🫁 Lung Cancer Prediction")
        st.markdown("#### Input Patient Details")
        gender = st.radio("Gender", ["M", "F"])
        age = st.number_input("Age (avg: 60)", min_value=0, max_value=120, value=60)

        st.markdown("#### Symptoms (check if present)")
        symptom_cols = [
            "Smoking", "Yellow Fingers", "Anxiety", "Peer Pressure", "Chronic Disease",
            "Fatigue", "Allergy", "Wheezing", "Alcohol", "Coughing",
            "Shortness of Breath", "Swallowing Difficulty", "Chest Pain"
        ]

        symptoms = [st.checkbox(sym) for sym in symptom_cols]

        if st.button("Predict Lung Cancer"):
            inputs = [1 if gender == "M" else 0, age] + [int(s) for s in symptoms]
            result, probability = predict("lung", inputs)
            st.write(f"Cancer Probability: {probability:.2%}")
            if result:
                st.error("🟥 Cancer Detected")
            else:
                st.success("✅ No Cancer Detected")

    elif cancer_type == "Skin":
        st.subheader("🩹 Skin Cancer Prediction")
        st.markdown("#### Input Patient Details")
        age = st.number_input("Age (avg: 40)", min_value=0, max_value=120, value=40)

        gender_options = ["male", "female", "unknown"]
        localization_options = [
            "scalp", "ear", "face", "back", "trunk", "chest", "upper extremity",
            "abdomen", "unknown", "lower extremity", "genital", "neck", "hand", "foot", "acral"
        ]
        dx_type_options = ["histo", "consensus", "confocal", "follow_up"]

        gender = st.selectbox("Gender", gender_options)
        localization = st.selectbox("Localization", localization_options)
        dx_type = st.selectbox("Diagnosis Type", dx_type_options)

        if st.button("Predict Skin Cancer"):
            result, probability = predict("skin", [age, gender, localization, dx_type])
            st.write(f"Cancer Probability: {probability:.2%}")
            if result:
                st.error("🟥 Cancer Detected")
            else:
                st.success("✅ No Cancer Detected")
