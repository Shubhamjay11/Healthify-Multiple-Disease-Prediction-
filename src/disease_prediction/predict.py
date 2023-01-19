import numpy as np
import config
import joblib


def get_disease(val):
    for key, value in config.disease_encode_dict.items():
        if val == value:
            return key


def predict_disease(inp_symptoms):
    test_lst = np.zeros(len(config.symptoms))
    for k in range(0, len(config.symptoms)):
        for s in inp_symptoms:
            if s == config.symptoms[k]:
                test_lst[k] = 1

    model = joblib.load(config.TRAINED_MODEL_RF)
    prediction = model.predict([test_lst])
    print()
    print(f"You have entered the following symptoms: {inp_symptoms}")
    print(f"You are suffering from : {get_disease(prediction[0])}")
    print("=" * 50)


if __name__ == "__main__":
    test_symptoms_1 = [
        "itching",
        "skin_rash",
        "stomach_pain",
        "abdominal_pain",
        "burning_micturition",
    ]
    predict_disease(test_symptoms_1)

    test_symptoms_2 = [
        "itching",
        "skin_rash",
        "nodal_skin_eruptions",
        "dischromic_patches",
    ]
    predict_disease(test_symptoms_2)

    test_symptoms_3 = [
        "vomiting",
        "weight_loss",
        "high_fever",
        "yellowish_skin",
        "fatigue",
    ]
    predict_disease(test_symptoms_3)