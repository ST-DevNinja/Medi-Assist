import streamlit as st
from ir_module.retriever import IRRetriever
from ai_module.a_star import graph, coordinates, disease_specialty, find_nearest_hospital
from streamlit_folium import st_folium
import folium
from ml_module.predictor import predict, predict_diabetes, predict_heart, predict_asthma
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
                scores = []
                for name, _, score in results:
                    scores.append(score)
                    retrieved_info += f"- `{name}` (Score: {score:.3f})\n"
                avg_score = sum(scores) / len(scores) if scores else 0.0
                retrieved_info += f"\n**📊 Evaluation Metric:** Average Similarity Score = `{avg_score:.3f}`"

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

    main_areas = [
        "Gulberg Lahore", "DHA Lahore", "Model Town Lahore", "Johar Town Lahore", "Shalimar Lahore",
        "Clifton Karachi", "Gulshan-e-Iqbal Karachi", "Korangi Karachi", "Defense Karachi", "Nazimabad Karachi",
        "Hayatabad Peshawar", "University Town Peshawar", "Durrani Peshawar", "Saddar Peshawar", "Karkhano Peshawar",
        "Satellite Town Quetta", "Quetta Cantt", "Sariab Quetta", "Chaman Quetta", "Killi Quetta",
        "Saddar, Rawalpindi", "Faizabad, Rawalpindi", "Chaklala, Rawalpindi", "Airport, Rawalpindi", "Bahria Town, Rawalpindi", "DHA Phase 2, Rawalpindi",
        "G-8, Islamabad", "G-10, Islamabad", "F-8, Islamabad", "Blue Area, Islamabad", "I-8, Islamabad",
        "Saddar", "Faizabad", "Chaklala", "Airport", "Bahria Town", "DHA Phase 2",
        "G-8", "G-10", "F-8", "Blue Area", "I-8"
    ]

    hospital_names = set(hospital_specialists.keys())

    # Filter only valid city locations that exist in coordinates and are not hospitals
    city_locations = sorted([
        loc for loc in coordinates.keys()
        if loc not in hospital_names and loc in main_areas
    ])

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

    from ml_module.predictor import predict, predict_diabetes, predict_heart

    disease_type = st.selectbox("Choose Disease Type", ["Diabetes", "Heart", "Kidney", "Liver", "Prostate", "Skin", "Asthma", "Lung"])

    if disease_type == "Diabetes":
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
            probability, accuracy = predict_diabetes(features)
            st.write(f"Diabetes Probability: {probability:.2%}")
            st.write(f"Model Accuracy: **{accuracy:.2%}**")

            if probability < 0.5:
                st.success("✅ Low Risk of Diabetes")
            elif 0.5 <= probability < 0.7:
                st.warning("⚠️ Moderate Risk of Diabetes (50% - 70%)")
            else:
                st.error("🟥 High Risk of Diabetes (>70%)")


    elif disease_type == "Heart":
        st.subheader("❤️ Heart Disease Prediction")
        st.markdown("#### Input Patient Details")   # <-- Changed heading here

        # Average values from Heart Disease dataset (approximate)
        avg_age = 55
        avg_sex = 1             # 1 = Male (most patients)
        avg_cp = 0              # Chest pain type (0: typical angina)
        avg_trestbps = 130      # Resting blood pressure
        avg_chol = 246          # Serum cholesterol
        avg_fbs = 0             # Fasting blood sugar > 120 mg/dl (0 = false)
        avg_restecg = 1         # Resting ECG results (mostly normal)
        avg_thalach = 150       # Max heart rate achieved
        avg_exang = 0           # Exercise induced angina (0 = no)
        avg_oldpeak = 1.0       # ST depression induced by exercise
        avg_slope = 1           # Slope of the peak exercise ST segment
        avg_ca = 0              # Number of major vessels (0-3)
        avg_thal = 2            # Thalassemia (2 = fixed defect)

        age = st.number_input("Age (Avg: 55)", min_value=1, max_value=120, value=avg_age)
        sex = st.radio("Gender (Male = 1, Female = 0)", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male", index=avg_sex)
        cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3], index=avg_cp)
        trestbps = st.number_input("Resting Blood Pressure (Avg: 130 mm Hg)", min_value=80, max_value=200, value=avg_trestbps)
        chol = st.number_input("Serum Cholesterol (Avg: 246 mg/dl)", min_value=100, max_value=600, value=avg_chol)
        fbs = st.radio("Fasting Blood Sugar > 120 mg/dl (1 = true; 0 = false)", [0, 1], index=avg_fbs)
        restecg = st.selectbox("Resting ECG results (0,1,2)", [0, 1, 2], index=avg_restecg)
        thalach = st.number_input("Max Heart Rate Achieved (Avg: 150)", min_value=60, max_value=220, value=avg_thalach)
        exang = st.radio("Exercise Induced Angina (1 = yes; 0 = no)", [0, 1], index=avg_exang)
        oldpeak = st.number_input("ST depression induced by exercise (Avg: 1.0)", min_value=0.0, max_value=10.0, value=avg_oldpeak)
        slope = st.selectbox("Slope of the peak exercise ST segment (0,1,2)", [0, 1, 2], index=avg_slope)
        ca = st.selectbox("Number of major vessels (0-3) colored by fluoroscopy", [0, 1, 2, 3], index=avg_ca)
        thal = st.selectbox("Thalassemia (1 = normal; 2 = fixed defect; 3 = reversible defect)", [1, 2, 3], index=avg_thal - 1)

        if st.button("Predict Heart Disease"):
            features = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
            result, probability, accuracy = predict_heart(features)
            st.write(f"Heart Disease Probability: {probability:.2%}")
            st.write(f"Model Accuracy: **{accuracy:.2%}**")

            if probability < 0.5:
                st.success("✅ Low Risk of Heart Disease")
            elif 0.5 <= probability < 0.7:
                st.warning("⚠️ Moderate Risk of Heart Disease (50% - 70%)")
            else:
                st.error("🟥 High Risk of Heart Disease (>70%)")


    
    
    elif disease_type == "Prostate":
        st.subheader("🧬 Prostate Cancer Prediction")
        st.markdown("#### Input Patient Details")
        age = st.number_input("Age (avg: 65)", min_value=0, max_value=120, value=65)
        psa = st.number_input("PSA Level (avg: 1.5 ng/mL)", value=1.5)
        bmi = st.number_input("BMI (avg: 26.5)", value=26.5)
        vol = st.number_input("Prostate Volume (avg: 35 cm³)", value=35.0)

        if st.button("Predict Prostate Cancer"):
            result, probability, accuracy = predict("prostate", [age, psa, bmi, vol])
            st.write(f"Prostate Cancer Probability: {probability:.2%}")
            st.write(f"Model Accuracy: **{accuracy:.2%}**")
            
            if result:
                if probability >= 0.7:
                    st.error("🟥 High Probability of Prostate Cancer Detected (>70%)")
                elif 0.5 <= probability < 0.7:
                    st.warning("⚠️ Moderate Probability of Prostate Cancer Detected (50% - 70%)")
                else:
                    st.info("ℹ️ Low Probability of Prostate Cancer, but still detected (<50%)")
            else:
                st.success("✅ No Prostate Cancer Detected")


    elif disease_type == "Lung":
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
            result, probability, accuracy = predict("lung", inputs)
            st.write(f"Cancer Probability: {probability:.2%}")
            st.write(f"Model Accuracy: **{accuracy:.2%}**")
            
            if result:
                st.error("🟥 Cancer Detected")
            else:
                st.success("✅ No Cancer Detected")

    elif disease_type == "Skin":
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
            result, probability, accuracy = predict("skin", [age, gender, localization, dx_type])
            st.write(f"Skin Cancer Probability: {probability:.2%}")
            st.write(f"Model Accuracy: **{accuracy:.2%}**")
            
            if probability < 0.5:
                st.success("✅ Low Risk of Skin Cancer")
            elif 0.5 <= probability < 0.7:
                st.warning("⚠️ Moderate Risk of Skin Cancer (50% - 70%)")
            else:
                st.error("🟥 High Risk of Skin Cancer (>70%)")


    
    elif disease_type == "Kidney":
        st.subheader("🍋 Kidney Disease Prediction")
        st.markdown("#### Input Patient Details")

        # Average values (estimates based on dataset)
        avg_age = 50
        avg_bp = 80
        avg_sg = 1.020
        avg_al = 1
        avg_su = 0
        avg_bgr = 121
        avg_bu = 43
        avg_sc = 1.2
        avg_sod = 137
        avg_pot = 4.6
        avg_hemo = 13.5

        age = st.number_input("Age", min_value=1, max_value=100, value=avg_age)
        bp = st.number_input("Blood Pressure (Avg: 80 mm Hg)", min_value=50, max_value=180, value=avg_bp)
        sg = st.selectbox("Specific Gravity (Avg: 1.020)", [1.005, 1.010, 1.015, 1.020, 1.025], index=2)
        al = st.selectbox("Albumin (Avg: 1)", [0, 1, 2, 3, 4, 5], index=avg_al)
        su = st.selectbox("Sugar (Avg: 0)", [0, 1, 2, 3, 4, 5], index=avg_su)
        bgr = st.number_input("Blood Glucose Random (Avg: 121)", min_value=50, max_value=500, value=avg_bgr)
        bu = st.number_input("Blood Urea (Avg: 43)", min_value=1, max_value=200, value=avg_bu)
        sc = st.number_input("Serum Creatinine (Avg: 1.2)", min_value=0.1, max_value=10.0, value=avg_sc)
        sod = st.number_input("Sodium (Avg: 137)", min_value=100, max_value=170, value=avg_sod)
        pot = st.number_input("Potassium (Avg: 4.6)", min_value=2.0, max_value=7.0, value=avg_pot)
        hemo = st.number_input("Hemoglobin (Avg: 13.5)", min_value=3.0, max_value=17.0, value=avg_hemo)

        # These need to be strings to match training encodings
        rbc = st.radio("Red Blood Cells", ["normal", "abnormal"])
        pc = st.radio("Pus Cell", ["normal", "abnormal"])
        pcc = st.radio("Pus Cell Clumps", ["notpresent", "present"])
        ba = st.radio("Bacteria", ["notpresent", "present"])
        htn = st.radio("Hypertension", ["no", "yes"])
        dm = st.radio("Diabetes Mellitus", ["no", "yes"])
        cad = st.radio("Coronary Artery Disease", ["no", "yes"])
        appet = st.radio("Appetite", ["good", "poor"])
        pe = st.radio("Pedal Edema", ["no", "yes"])
        ane = st.radio("Anemia", ["no", "yes"])
        pcv = st.text_input("Packed Cell Volume (as number)", "40")
        wc = st.text_input("White Blood Cell Count (as number)", "8000")
        rc = st.text_input("Red Blood Cell Count (as float)", "5.2")

        if st.button("Predict Kidney Disease"):
            try:
                # Ensure numeric values are cast properly
                pcv = float(pcv)
                wc = float(wc)
                rc = float(rc)

                features = [
                    sg, al, su, rbc, pc, pcc, ba, bgr, bu, sc, sod, pot,
                    hemo, pcv, wc, rc, htn, dm, cad, appet, pe, ane
                ]

                result, probability, accuracy = predict("kidney", features)
                st.write(f"Kidney Disease Probability: {probability:.2%}")
                st.write(f"Model Accuracy: **{accuracy:.2%}**")
                
                if probability < 0.5:
                    st.success("✅ Low Risk of Kidney Disease")
                elif 0.5 <= probability < 0.7:
                    st.warning("⚠️ Moderate Risk of Kidney Disease (50% - 70%)")
                else:
                    st.error("🟥 High Risk of Kidney Disease (>70%)")

            except ValueError:
                st.error("Please enter valid numeric values for PCV, WBC, and RBC counts.")



    elif disease_type == "Liver":
        st.subheader("🟤 Liver Disease Prediction")
        st.markdown("#### Input Patient Details")

        # Sample average values based on ILPD dataset
        age = st.number_input("Age (avg: 45)", min_value=1, max_value=120, value=45)
        gender = st.selectbox("Gender", options=["Male", "Female"])
        total_bilirubin = st.number_input("Total Bilirubin (avg: 1.0)", min_value=0.0, value=1.0)
        direct_bilirubin = st.number_input("Direct Bilirubin (avg: 0.3)", min_value=0.0, value=0.3)
        alk_phos = st.number_input("Alkaline Phosphotase (avg: 200)", min_value=50, value=200)
        sgpt = st.number_input("SGPT (avg: 80)", min_value=10, value=80)
        sgot = st.number_input("SGOT (avg: 90)", min_value=10, value=90)
        total_protein = st.number_input("Total Proteins (avg: 6.5)", min_value=0.0, value=6.5)
        albumin = st.number_input("Albumin (avg: 3.5)", min_value=0.0, value=3.5)
        ag_ratio = st.number_input("Albumin/Globulin Ratio (avg: 1.0)", min_value=0.0, value=1.0)

        if st.button("Predict Liver Disease"):
            gender_num = 1 if gender == "Male" else 0
            features = [age, gender_num, total_bilirubin, direct_bilirubin, alk_phos,
                        sgpt, sgot, total_protein, albumin, ag_ratio]
            result, probability, accuracy = predict("liver", features)
            st.write(f"Liver Disease Probability: {probability:.2%}")
            st.write(f"Model Accuracy: **{accuracy:.2%}**")
            
            if probability < 0.5:
                st.success("✅ Liver is Healthy")
            elif 0.5 <= probability < 0.7:
                st.warning("⚠️ Moderate risk of Liver Disease (50% - 70%)")
            else:  # probability >= 0.7
                st.error("🟥 High risk of Liver Disease (>70%)")



    elif disease_type == "Asthma":
        st.subheader("🌬️ Asthma Prediction")
        st.markdown("#### 🫁 Input Patient Details")

        # Suggested average/default values
        avg_tiredness = 0
        avg_dry_cough = 1
        avg_difficulty_breath = 1
        avg_sore_throat = 0
        avg_none_sympton = 0

        avg_pains = 0
        avg_nasal_congestion = 1
        avg_runny_nose = 1
        avg_none_experiencing = 0

        avg_age = 30

        avg_gender = "Male"  # or "Female" depending on your dataset

        avg_severity_mild = 0
        avg_severity_moderate = 0

        
        # Age input and bucket one-hot encoding
        age = st.number_input("Age", min_value=0, max_value=120, value=avg_age)

        age_0_9 = 1 if age <= 9 else 0
        age_10_19 = 1 if 10 <= age <= 19 else 0
        age_20_24 = 1 if 20 <= age <= 24 else 0
        age_25_59 = 1 if 25 <= age <= 59 else 0
        age_60_plus = 1 if age >= 60 else 0

        
        # Symptom inputs (binary yes/no)
        tiredness = st.radio("Tiredness?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", index=avg_tiredness)
        dry_cough = st.radio("Dry Cough?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", index=avg_dry_cough)
        difficulty_breath = st.radio("Difficulty in Breathing?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", index=avg_difficulty_breath)
        sore_throat = st.radio("Sore Throat?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", index=avg_sore_throat)
        none_sympton = st.radio("No Symptoms?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", index=avg_none_sympton)

        pains = st.radio("Pains?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", index=avg_pains)
        nasal_congestion = st.radio("Nasal Congestion?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", index=avg_nasal_congestion)
        runny_nose = st.radio("Runny Nose?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", index=avg_runny_nose)
        none_experiencing = st.radio("None Experiencing?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", index=avg_none_experiencing)

        # Gender one-hot encoding
        gender = st.radio("Gender", ["Female", "Male"], index=1 if avg_gender == "Male" else 0)
        gender_female = 1 if gender == "Female" else 0
        gender_male = 1 if gender == "Male" else 0

        # Severity inputs
        severity_mild = st.radio("Severity Mild?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", index=avg_severity_mild)
        severity_moderate = st.radio("Severity Moderate?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", index=avg_severity_moderate)

        if st.button("Predict Asthma"):
            features = [
                tiredness, dry_cough, difficulty_breath, sore_throat, none_sympton,
                pains, nasal_congestion, runny_nose, none_experiencing,
                age_0_9, age_10_19, age_20_24, age_25_59, age_60_plus,
                gender_female, gender_male,
                severity_mild, severity_moderate
            ]

            result, probability, accuracy = predict_asthma(features)
            st.write(f"Asthma Probability: {probability:.2%}")
            st.write(f"Model Accuracy: **{accuracy:.2%}**")

            if result:
                if probability >= 0.7:
                    st.error("🟥 High Risk of Asthma Detected (>70%)")
                elif 0.5 <= probability < 0.7:
                    st.warning("⚠️ Moderate Risk of Asthma Detected (50% - 70%)")
                else:
                    st.info("ℹ️ Low Probability of Asthma, but still detected (<50%)")
            else:
                st.success("✅ No Asthma Detected")
