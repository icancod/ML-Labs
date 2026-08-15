
import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Care Match",
    page_icon="🩺",
    layout="wide"
)

# ============================================================
# LOAD TRAINED ML COMPONENTS
# ============================================================

tfidf = joblib.load(
    "streamlit_models/tfidf_vectorizer.pkl"
)

label_encoder = joblib.load(
    "streamlit_models/label_encoder.pkl"
)

lr_model = joblib.load(
    "streamlit_models/logistic_regression.pkl"
)

rf_model = joblib.load(
    "streamlit_models/random_forest.pkl"
)

xgb_model = joblib.load(
    "streamlit_models/xgboost.pkl"
)

# ============================================================
# LOAD REAL DOCTOR AND HOSPITAL DATASETS
# ============================================================

doctors = pd.read_csv(
    "bangalore_doctors_final.csv"
)

hospitals = pd.read_csv(
    "hospital_data_bangalore.csv"
)

# ============================================================
# DISEASE → DOCTOR SPECIALTY MAPPING
# ============================================================

DISEASE_SPECIALTY = {

    "Acne": "dermatologist",

    "Arthritis": "rheumatologist",

    "Bronchial Asthma": "pulmonologist",

    "Cervical spondylosis": "orthopedist",

    "Chicken pox": "dermatologist",

    "Common Cold": "general-physician",

    "Dengue": "general-physician",

    "Dimorphic Hemorrhoids": "surgeon",

    "Fungal infection": "dermatologist",

    "Hypertension": "cardiologist",

    "Impetigo": "dermatologist",

    "Jaundice": "gastroenterologist",

    "Malaria": "general-physician",

    "Migraine": "neurologist",

    "Pneumonia": "pulmonologist",

    "Psoriasis": "dermatologist",

    "Typhoid": "general-physician",

    "Varicose Veins": "vascular-surgeon",

    "allergy": "general-physician",

    "diabetes": "endocrinologist",

    "drug reaction": "general-physician",

    "gastroesophageal reflux disease": "gastroenterologist",

    "peptic ulcer disease": "gastroenterologist",

    "urinary tract infection": "urologist"
}

# ============================================================
# SYMPTOM VOCABULARY
# ============================================================

SYMPTOMS = [
    "abdominal pain",
    "back pain",
    "chest pain",
    "chills",
    "constipation",
    "cough",
    "diarrhea",
    "dizziness",
    "fatigue",
    "fever",
    "headache",
    "heartburn",
    "indigestion",
    "itching",
    "joint pain",
    "muscle pain",
    "nausea",
    "nasal congestion",
    "runny nose",
    "rash",
    "shortness of breath",
    "sore throat",
    "stomach pain",
    "vomiting",
    "wheezing",
    "weakness",
    "weight loss",
    "weight gain",
    "loss of appetite",
    "increased appetite",
    "frequent urination",
    "painful urination",
    "blood in urine",
    "dark urine",
    "yellow skin",
    "blurred vision",
    "vision problems",
    "swelling",
    "skin peeling",
    "dry skin",
    "red skin",
    "scaly skin",
    "sweating",
    "mucus",
    "phlegm",
    "sneezing",
    "blocked nose",
    "difficulty swallowing",
    "bloating",
    "gas",
    "acid reflux",
    "burning sensation",
    "palpitations",
    "neck pain",
    "stiffness",
    "tremors",
    "balance problems",
    "memory problems",
    "difficulty concentrating",
    "sleep problems",
    "anxiety",
    "depression"
]

# ============================================================
# HEADER
# ============================================================

st.title("🩺 Care Match")

st.subheader(
    "AI-Powered Symptom-Based Disease Prediction"
)

st.write(
    "Select the symptoms you are experiencing. "
    "The selected symptoms will be processed by the "
    "trained machine-learning models."
)

st.divider()

# ============================================================
# SYMPTOM SELECTION
# ============================================================

selected_symptoms = st.multiselect(
    "Select your symptoms:",
    options=SYMPTOMS,
    help="You can select multiple symptoms."
)

# ============================================================
# PREDICTION
# ============================================================

if st.button(
    "Predict Disease",
    type="primary"
):

    if len(selected_symptoms) == 0:

        st.warning(
            "Please select at least one symptom."
        )

    else:

        # ----------------------------------------------------
        # Convert selected symptoms to text
        # ----------------------------------------------------

        symptom_text = " ".join(
            selected_symptoms
        )

        # ----------------------------------------------------
        # TF-IDF transformation
        # ----------------------------------------------------

        X_input = tfidf.transform(
            [symptom_text]
        )

        # ----------------------------------------------------
        # Individual model probabilities
        # ----------------------------------------------------

        lr_prob = lr_model.predict_proba(
            X_input
        )[0]

        rf_prob = rf_model.predict_proba(
            X_input
        )[0]

        xgb_prob = xgb_model.predict_proba(
            X_input
        )[0]

        # ----------------------------------------------------
        # Individual predictions
        # ----------------------------------------------------

        lr_prediction = np.argmax(
            lr_prob
        )

        rf_prediction = np.argmax(
            rf_prob
        )

        xgb_prediction = np.argmax(
            xgb_prob
        )

        # ----------------------------------------------------
        # WEIGHTED SOFT VOTING
        #
        # Logistic Regression = 2
        # Random Forest       = 2
        # XGBoost             = 1
        # ----------------------------------------------------

        weighted_probability = (
            2 * lr_prob +
            2 * rf_prob +
            1 * xgb_prob
        ) / 5

        final_prediction = np.argmax(
            weighted_probability
        )

        final_disease = (
            label_encoder.inverse_transform(
                [final_prediction]
            )[0]
        )

        final_probability = (
            weighted_probability[
                final_prediction
            ] * 100
        )

        # ====================================================
        # SELECTED SYMPTOMS
        # ====================================================

        st.divider()

        st.subheader(
            "Selected Symptoms"
        )

        st.write(
            ", ".join(selected_symptoms)
        )

        # ====================================================
        # INDIVIDUAL MODEL PREDICTIONS
        # ====================================================

        st.subheader(
            "Individual Model Predictions"
        )

        prediction_data = pd.DataFrame({

            "Model": [
                "Logistic Regression",
                "Random Forest",
                "XGBoost"
            ],

            "Prediction": [

                label_encoder.inverse_transform(
                    [lr_prediction]
                )[0],

                label_encoder.inverse_transform(
                    [rf_prediction]
                )[0],

                label_encoder.inverse_transform(
                    [xgb_prediction]
                )[0]
            ]
        })

        st.dataframe(
            prediction_data,
            hide_index=True,
            use_container_width=True
        )

        # ====================================================
        # FINAL ENSEMBLE PREDICTION
        # ====================================================

        st.subheader(
            "Final Ensemble Prediction"
        )

        st.success(
            f"Predicted Disease: {final_disease}"
        )

        st.metric(
            "Weighted Model Probability",
            f"{final_probability:.2f}%"
        )

        # ====================================================
        # TOP DISEASE PREDICTIONS
        # ====================================================

        st.subheader(
            "Top Disease Predictions"
        )

        probability_table = pd.DataFrame({

            "Disease":
                label_encoder.classes_,

            "Weighted Probability":
                weighted_probability
        })

        probability_table = (
            probability_table
            .sort_values(
                "Weighted Probability",
                ascending=False
            )
            .head(10)
            .reset_index(drop=True)
        )

        probability_table[
            "Weighted Probability"
        ] = (
            probability_table[
                "Weighted Probability"
            ] * 100
        ).round(2)

        st.dataframe(
            probability_table,
            hide_index=True,
            use_container_width=True
        )

        # ====================================================
        # DOCTOR RECOMMENDATION
        # ====================================================

        st.divider()

        st.header(
            "👨‍⚕️ Doctor Recommendation"
        )

        recommended_specialty = (
            DISEASE_SPECIALTY.get(
                final_disease
            )
        )

        if recommended_specialty:

            st.write(
                f"Recommended specialty: "
                f"**{recommended_specialty.replace('-', ' ').title()}**"
            )

            matching_doctors = doctors[
                doctors["specialty"].str.lower()
                == recommended_specialty.lower()
            ].copy()

            if len(matching_doctors) > 0:

                # --------------------------------------------
                # Rank doctors
                # --------------------------------------------

                matching_doctors = (
                    matching_doctors
                    .sort_values(
                        by=[
                            "rating",
                            "experience_years"
                        ],
                        ascending=[
                            False,
                            False
                        ]
                    )
                    .head(3)
                )

                st.write(
                    f"Top {len(matching_doctors)} "
                    f"matching doctors:"
                )

                for _, doctor in (
                    matching_doctors.iterrows()
                ):

                    with st.container(
                        border=True
                    ):

                        st.subheader(
                            doctor["name"]
                        )

                        st.write(
                            f"**Specialty:** "
                            f"{doctor['specialty'].replace('-', ' ').title()}"
                        )

                        st.write(
                            f"**Degree:** "
                            f"{doctor['degree']}"
                        )

                        st.write(
                            f"**Experience:** "
                            f"{doctor['experience_years']} years"
                        )

                        st.write(
                            f"**Rating:** ⭐ "
                            f"{doctor['rating']}"
                        )

                        st.write(
                            f"**Consultation Fee:** "
                            f"₹{doctor['consultation_fee']}"
                        )

                        st.write(
                            f"**Location:** "
                            f"{doctor['bangalore_location']}"
                        )

                        if pd.notna(
                            doctor["google_maps_link"]
                        ):

                            st.link_button(
                                "📍 View on Google Maps",
                                doctor[
                                    "google_maps_link"
                                ]
                            )

            else:

                st.info(
                    "No doctors were found for "
                    "the recommended specialty."
                )

        # ====================================================
            # HOSPITAL RECOMMENDATION
            # ====================================================

            st.divider()

            st.header(
                "🏥 Hospital Recommendation"
            )

            st.info(
                "Hospital recommendations are based on "
                "the available Bangalore hospital dataset. "
                "The dataset does not contain disease-specific "
                "hospital specialties, so hospitals are ranked "
                "using rating and number of ratings rather than "
                "claiming disease-specific specialization."
            )

            # ----------------------------------------------------
            # Convert number-of-ratings to numeric score
            # ----------------------------------------------------

            hospital_display = hospitals.copy()

            def parse_rating_count(value):

                if pd.isna(value):
                    return 0

                value = str(value)

                # Remove quotes, parentheses and spaces
                value = (
                    value
                    .replace("'", "")
                    .replace("(", "")
                    .replace(")", "")
                    .replace(",", "")
                    .strip()
                )

                # Values such as 8.6T
                if value.lower().endswith("t"):

                    try:
                        return float(
                            value[:-1]
                        ) * 1000

                    except:
                        return 0

                try:
                    return float(value)

                except:
                    return 0

            hospital_display[
                "rating_count_numeric"
            ] = hospital_display[
                "No_of_people_rated"
            ].apply(
                parse_rating_count
            )

           # ====================================================
    # HOSPITAL RECOMMENDATION
    # ====================================================

    st.divider()

    st.header("🏥 Hospital Recommendation")

    # ====================================================
    # DISEASE-SPECIFIC HOSPITAL KEYWORDS
    # ====================================================

    DISEASE_HOSPITAL_KEYWORDS = {

        "Acne": [
            "dermatology",
            "skin",
            "dermatologist"
        ],

        "Arthritis": [
            "orthopedic",
            "orthopaedic",
            "rheumatology",
            "rheumatologist",
            "joint"
        ],

        "Bronchial Asthma": [
            "pulmonology",
            "pulmonary",
            "respiratory",
            "chest",
            "lung"
        ],

        "Cervical spondylosis": [
            "orthopedic",
            "orthopaedic",
            "spine",
            "neurology"
        ],

        "Chicken pox": [
            "dermatology",
            "skin",
            "general physician",
            "general medicine"
        ],

        "Common Cold": [
            "general physician",
            "general medicine",
            "internal medicine"
        ],

        "Dengue": [
            "general physician",
            "general medicine",
            "infectious",
            "internal medicine"
        ],

        "Dimorphic Hemorrhoids": [
            "proctology",
            "colorectal",
            "general surgery",
            "surgery"
        ],

        "Fungal infection": [
            "dermatology",
            "skin"
        ],

        "Hypertension": [
            "cardiology",
            "cardiac",
            "heart",
            "hypertension",
            "cardiovascular",
            "vascular",
            "general medicine"
        ],

        "Impetigo": [
            "dermatology",
            "skin"
        ],

        "Jaundice": [
            "gastroenterology",
            "gastro",
            "liver",
            "hepatology"
        ],

        "Malaria": [
            "general physician",
            "general medicine",
            "infectious",
            "internal medicine"
        ],

        "Migraine": [
            "neurology",
            "neurologist",
            "neurosurgery"
        ],

        "Pneumonia": [
            "pulmonology",
            "pulmonary",
            "respiratory",
            "chest",
            "lung"
        ],

        "Psoriasis": [
            "dermatology",
            "skin"
        ],

        "Typhoid": [
            "general physician",
            "general medicine",
            "infectious",
            "internal medicine"
        ],

        "Varicose Veins": [
            "vascular",
            "vascular surgery",
            "vein",
            "cardiovascular"
        ],

        "allergy": [
            "allergy",
            "immunology",
            "general physician",
            "general medicine"
        ],

        "diabetes": [
            "diabetes",
            "endocrinology",
            "endocrine",
            "general medicine"
        ],

        "drug reaction": [
            "dermatology",
            "skin",
            "allergy",
            "immunology",
            "general medicine",
            "general physician",
            "internal medicine"
        ],

        "gastroesophageal reflux disease": [
            "gastroenterology",
            "gastro",
            "digestive"
        ],

        "peptic ulcer disease": [
            "gastroenterology",
            "gastro",
            "digestive",
            "general surgery"
        ],

        "urinary tract infection": [
            "urology",
            "urologist",
            "kidney",
            "urinary"
        ]
    }


    # ====================================================
    # CLEAN HOSPITAL DATA
    # ====================================================

    hospital_display = hospitals.copy()

    # Convert important fields to strings
    hospital_display["Hospital_name"] = (
        hospital_display["Hospital_name"]
        .fillna("")
        .astype(str)
    )

    hospital_display["Type"] = (
        hospital_display["Type"]
        .fillna("")
        .astype(str)
    )

    hospital_display["Address"] = (
        hospital_display["Address"]
        .fillna("")
        .astype(str)
    )

    hospital_display["Highlighted_review"] = (
        hospital_display["Highlighted_review"]
        .fillna("")
        .astype(str)
    )

    # ====================================================
    # CLEAN RATING
    # ====================================================

    hospital_display["Rating_numeric"] = pd.to_numeric(
        hospital_display["Rating"],
        errors="coerce"
    )

    # Remove obviously invalid ratings
    hospital_display = hospital_display[
        hospital_display["Rating_numeric"].between(
            0,
            5
        )
    ].copy()


    # ====================================================
    # CLEAN NUMBER OF REVIEWS
    # ====================================================

    def parse_review_count(value):

        if pd.isna(value):
            return 0

        value = str(value)

        # Remove unwanted characters
        value = (
            value
            .replace("'", "")
            .replace('"', "")
            .replace("(", "")
            .replace(")", "")
            .replace(",", "")
            .strip()
        )

        if value == "":
            return 0

        # Handle values such as 8.6T
        if value.lower().endswith("t"):

            try:
                return float(
                    value[:-1]
                ) * 1000

            except:
                return 0

        try:
            return float(value)

        except:
            return 0


    hospital_display["Review_count_numeric"] = (
        hospital_display["No_of_people_rated"]
        .apply(parse_review_count)
    )


    # ====================================================
    # CREATE SEARCH TEXT
    # ====================================================

    hospital_display["search_text"] = (

        hospital_display["Hospital_name"]
        + " "
        + hospital_display["Type"]
        + " "
        + hospital_display["Address"]
        + " "
        + hospital_display["Highlighted_review"]

    ).str.lower()


    # ====================================================
    # GET KEYWORDS FOR PREDICTED DISEASE
    # ====================================================

    hospital_keywords = DISEASE_HOSPITAL_KEYWORDS.get(
        final_disease,
        []
    )


    # ====================================================
    # CALCULATE DISEASE RELEVANCE
    # ====================================================

    def calculate_hospital_relevance(text):

        score = 0

        for keyword in hospital_keywords:

            if keyword.lower() in text:
                score += 1

        return score


    hospital_display["relevance_score"] = (
        hospital_display["search_text"]
        .apply(
            calculate_hospital_relevance
        )
    )


    # ====================================================
    # KEEP ONLY DISEASE-RELEVANT HOSPITALS
    # ====================================================

    relevant_hospitals = hospital_display[
        hospital_display["relevance_score"] > 0
    ].copy()


    # ====================================================
    # RANK RELEVANT HOSPITALS
    # ====================================================

    if len(relevant_hospitals) > 0:

        # Normalize rating
        relevant_hospitals["rating_score"] = (
            relevant_hospitals["Rating_numeric"] / 5
        )

        # Log-transform review count so huge review counts
        # don't dominate the ranking
        relevant_hospitals["review_score"] = np.log1p(
            relevant_hospitals["Review_count_numeric"]
        )

        max_review = (
            relevant_hospitals["review_score"].max()
        )

        if max_review > 0:

            relevant_hospitals["review_score"] = (
                relevant_hospitals["review_score"]
                / max_review
            )

        # Final hospital recommendation score
        relevant_hospitals["hospital_score"] = (

            0.50
            * relevant_hospitals["relevance_score"]

            + 0.35
            * relevant_hospitals["rating_score"]

            + 0.15
            * relevant_hospitals["review_score"]

        )

        relevant_hospitals = (
            relevant_hospitals
            .sort_values(
                by=[
                    "hospital_score",
                    "Rating_numeric",
                    "Review_count_numeric"
                ],
                ascending=[
                    False,
                    False,
                    False
                ]
            )
            .head(5)
        )


    # ====================================================
    # DISPLAY RECOMMENDATIONS
    # ====================================================

    if len(relevant_hospitals) == 0:

        st.warning(
            f"No hospital in the dataset was found with "
            f"keywords related to **{final_disease}**."
        )

        st.info(
            "We are not displaying unrelated hospitals. "
            "This avoids recommending a hospital simply "
            "because it has a high rating."
        )

    else:

        st.success(
            f"Hospital recommendations relevant to "
            f"**{final_disease}**"
        )

        st.caption(
            "Hospitals are ranked using disease-related "
            "keyword relevance, rating, and number of reviews "
            "available in the dataset."
        )

        for _, hospital in relevant_hospitals.iterrows():

            with st.container(border=True):

                st.subheader(
                    hospital["Hospital_name"]
                )

                st.write(
                    f"**Disease relevance:** "
                    f"{hospital['relevance_score']} matching keyword(s)"
                )

                st.write(
                    f"**Rating:** ⭐ "
                    f"{hospital['Rating_numeric']:.1f}"
                )

                review_count = (
                    hospital["Review_count_numeric"]
                )

                if review_count > 0:

                    st.write(
                        f"**People Rated:** "
                        f"{int(review_count):,}"
                    )

                hospital_type = hospital["Type"].strip("'")
                st.write(
                    f"**Type:** {hospital_type}"
                )

                address = hospital["Address"]

                if (
                    address
                    and address.lower() != "nan"
                ):

                    st.write(
                        f"**Address:** {address}"
                    )

                phone = hospital["Phone_number"]

                if (
                    pd.notna(phone)
                    and str(phone).strip() != ""
                    and str(phone).lower() != "nan"
                ):

                    st.write(
                        f"**Phone:** {phone}"
                    )
            # ====================================================
            # DISCLAIMER
            # ====================================================

            st.divider()

        st.warning(
            "⚠️ This application is an AI-based "
            "decision-support prototype and is not "
            "a medical diagnosis. Predictions and "
            "recommendations should be reviewed by "
            "a qualified healthcare professional."
        )
