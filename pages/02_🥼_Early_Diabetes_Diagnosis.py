import config as cfg
import joblib
import numpy as np
import streamlit as st
from config import doctor_search


def predict_early_diabetes(
    age,
    gender,
    polyuria,
    polydipsia,
    weight,
    weakness,
    polyphagia,
    genital_thrush,
    visual_blurring,
    itching,
    irritability,
    delayed_healing,
    partial_paresis,
    muscle_stiffness,
    alopecia,
    obesity,
):
    model = joblib.load(cfg.DIABETES_MODEL)
    prediction = model.predict(
        np.array(
            [
                [
                    int(age),
                    int(gender),
                    int(polyuria),
                    int(polydipsia),
                    int(weight),
                    int(weakness),
                    int(polyphagia),
                    int(genital_thrush),
                    int(visual_blurring),
                    int(itching),
                    int(irritability),
                    int(delayed_healing),
                    int(partial_paresis),
                    int(muscle_stiffness),
                    int(alopecia),
                    int(obesity),
                ]
            ]
        )
    )
    return prediction


def diabetes_app():
    st.set_page_config(
        page_title="Healthify - Diabetes Diagnosis",
        page_icon="🏥",
    )
    st.markdown(
        f"<h1 style='text-align: center; color: black;'>Early Diabetes Diagnosis</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<h4 style='text-align: center; color: black;'>Choose below options according to the report to know the patient's status</h4>",
        unsafe_allow_html=True,
    )

    age = st.slider("What is your Age?", 0, 110, 25)

    gender = st.radio("What is your Gender?", ("Male", "Female"))
    if gender == "Male":
        gender = 1
    else:
        gender = 0

    polyuria = st.radio("Do you have Polyuria?", ("Yes", "No"))
    if polyuria == "Yes":
        polyuria = 1
    else:
        polyuria = 0
    link1 = "[what is Polyuria?](https://en.wikipedia.org/wiki/Polyuria)"
    st.markdown(link1, unsafe_allow_html=True)

    polydipsia = st.radio("Do you have Polydipsia?", ("Yes", "No"))
    if polydipsia == "Yes":
        polydipsia = 1
    else:
        polydipsia = 0
    link2 = "[what is Polydipsia?](https://en.wikipedia.org/wiki/Polydipsia)"
    st.markdown(link2, unsafe_allow_html=True)

    weight = st.radio("Recently do you observe sudden weight loss?", ("Yes", "No"))
    if weight == "Yes":
        weight = 1
    else:
        weight = 0

    weakness = st.radio("Do you feel any Weakness?", ("Yes", "No"))
    if weakness == "Yes":
        weakness = 1
    else:
        weakness = 0

    polyphagia = st.radio("Do you have Polyphagia?", ("Yes", "No"))
    if polyphagia == "Yes":
        polyphagia = 1
    else:
        polyphagia = 0
    link3 = "[what is Polyphagia?](https://en.wikipedia.org/wiki/Polyphagia)"
    st.markdown(link3, unsafe_allow_html=True)

    genital_thrush = st.radio("Do you have Genital thrush?", ("Yes", "No"))
    if genital_thrush == "Yes":
        genital_thrush = 1
    else:
        genital_thrush = 0
    link4 = "[what is Genital thrush?](https://www.ticahealth.org/interactive-guide/your-body/genital-problems/genital-thrush/)"
    st.markdown(link4, unsafe_allow_html=True)

    visual_blurring = st.radio("Do you have Visual blurring?", ("Yes", "No"))
    if visual_blurring == "Yes":
        visual_blurring = 1
    else:
        visual_blurring = 0
    link5 = "[what is Visua blurring?](https://en.wikipedia.org/wiki/Blurred_vision)"
    st.markdown(link5, unsafe_allow_html=True)

    itching = st.radio("Do you have Itching?", ("Yes", "No"))
    if itching == "Yes":
        itching = 1
    else:
        itching = 0
    link6 = "[what is Itching?](https://en.wikipedia.org/wiki/Itch)"
    st.markdown(link6, unsafe_allow_html=True)

    irritability = st.radio("Do you have Irritability?", ("Yes", "No"))
    if irritability == "Yes":
        irritability = 1
    else:
        irritability = 0
    link7 = "[what is Irritability?](https://en.wikipedia.org/wiki/Irritability)"
    st.markdown(link7, unsafe_allow_html=True)

    delayed_healing = st.radio("Do you have Delayed healing?", ("Yes", "No"))
    if delayed_healing == "Yes":
        delayed_healing = 1
    else:
        delayed_healing = 0

    partial_paresis = st.radio("Do you have Partial paresis?", ("Yes", "No"))
    if partial_paresis == "Yes":
        partial_paresis = 1
    else:
        partial_paresis = 0
    link8 = "[what is Paresis?](https://en.wikipedia.org/wiki/Paresis)"
    st.markdown(link8, unsafe_allow_html=True)

    muscle_stiffness = st.radio("Do you have Muscle stiffness?", ("Yes", "No"))
    if muscle_stiffness == "Yes":
        muscle_stiffness = 1
    else:
        muscle_stiffness = 0

    alopecia = st.radio("Do you have Alopecia?", ("Yes", "No"))
    if alopecia == "Yes":
        alopecia = 1
    else:
        alopecia = 0
    link9 = "[what is Alopecia?](https://en.wikipedia.org/wiki/Hair_loss)"
    st.markdown(link9, unsafe_allow_html=True)

    obesity = st.radio("Do you have Obesity?", ("Yes", "No"))
    if obesity == "Yes":
        obesity = 1
    else:
        obesity = 0

    result = None
    if st.button("Predict"):
        result = predict_early_diabetes(
            age,
            gender,
            polyuria,
            polydipsia,
            weight,
            weakness,
            polyphagia,
            genital_thrush,
            visual_blurring,
            itching,
            irritability,
            delayed_healing,
            partial_paresis,
            muscle_stiffness,
            alopecia,
            obesity,
        )
        if result == 1:
            st.subheader("The patient have high chances of having Diabetes 😔")

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
            st.markdown(f"- [Endocrinologist]({infec}) 👨‍⚕")
            st.markdown("---")
        else:
            st.subheader("The patient does not have Diabetes 😄")


if __name__ == "__main__":
    diabetes_app()
