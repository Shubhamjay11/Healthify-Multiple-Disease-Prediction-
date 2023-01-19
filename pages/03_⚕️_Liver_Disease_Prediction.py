import config as cfg
import joblib
import numpy as np
import streamlit as st
from config import doctor_search


def liver_app():
    st.set_page_config(
        page_title="Healthify - Liver Disease Prediction",
        page_icon="🏥",
    )
    st.markdown(
        f"<h1 style='text-align: center; color: black;'>Liver Disease Diagnosis</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<h4 style='text-align: center; color: black;'>Choose below options according to the report to know the patient's status</h4>",
        unsafe_allow_html=True,
    )

    model = joblib.load(cfg.LIVER_MODEL)

    gender = st.radio("Gender", ("Male", "Female"))

    if gender == "Male":
        gender = 1
    else:
        gender = 0
    age = st.slider("Age", min_value=1, max_value=110)

    total_bilirubin = st.slider(
        "Total Bilirubin (in mg/dL)", min_value=0.1, max_value=40.0
    )

    direct_bilirubin = st.slider("Direct Bilirubin (in mg/dL)")

    alk_phosphate = st.slider(
        "Alkaline Phosphotase (in units/L", min_value=10, max_value=250
    )

    alamine_aminotransferase = st.slider(
        "Alamine Aminotransferase (in units/L)", min_value=20, max_value=120
    )

    aspartate_aminotransferase = st.slider(
        "Aspartate Aminotransferase (in units/L)", min_value=20, max_value=140
    )

    total_proteins = st.slider(
        "Total Proteins (in g/dL)", min_value=2.0, max_value=10.0, step=0.1
    )

    albumin = st.slider("Albumin (in g/dL)", min_value=1.0, max_value=10.0, step=0.1)

    ratio = st.slider(
        "Albumin and Globulin Ratio", min_value=0.1, max_value=3.0, step=0.01
    )

    inp_array = np.array(
        [
            [
                age,
                gender,
                total_bilirubin,
                direct_bilirubin,
                alk_phosphate,
                alamine_aminotransferase,
                aspartate_aminotransferase,
                total_proteins,
                albumin,
                ratio,
            ]
        ]
    )

    predict = st.button("Predict")
    if predict:
        liver_disease_prob = model.predict(inp_array)

        if liver_disease_prob == 1:
            st.subheader("The patient have chances of having a liver disease 😔")
            st.markdown("---")
            st.error(
                "If you are a patient, consult with one of the following doctors immediately"
            )
            st.subheader("Specialists 👨‍⚕")

            st.write(
                "Click on the specialists to get the specialists nearest to your location 📍"
            )
            pcp = doctor_search("Primary Care Provider")
            infec = doctor_search("Endocrinologist")
            st.markdown(f"- [Primary Care Doctor]({pcp}) 👨‍⚕")
            st.markdown(f"- [Gastroenterologists]({infec}) 👨‍⚕")
            st.markdown("---")
        if liver_disease_prob == 2:

            st.subheader(
                "The patient doesn't have any chances of having a liver disease 😄"
            )

            st.balloons()


if __name__ == "__main__":
    liver_app()
