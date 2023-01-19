import config as cfg
import streamlit as st
import torch
import torch.nn as nn
import torchvision
from config import doctor_search
from PIL import Image
from torchvision import transforms


def preprocess_image(image):
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ]
    )
    image = Image.open(image)
    image = transform(image).float()
    return image


class PneumoniaModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = torchvision.models.resnet50(pretrained=False)
        self.network.conv1 = nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        for param in self.network.fc.parameters():
            param.require_grad = False

        num_features = (
            self.network.fc.in_features
        )  # get number of features of last layer
        # -----------------------------------
        self.network.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
        )

    def forward(self, x):
        return self.network(x)


def predict(image):
    with torch.no_grad():

        image = preprocess_image(image)
        image = image.unsqueeze(0)
        output = model(image)
        softmax = nn.Softmax(dim=1)
        output = softmax(output)
        pred = torch.argmax(output)
        print(output)
        confidence = torch.max(output).numpy()
        print(confidence * 100)
    return pred, confidence


def app():
    st.set_page_config(
        page_title="Healthify - Pneumonia Detection",
        page_icon="🏥",
    )
    st.markdown(
        f"<h1 style='text-align: center; color: black;'>Pneumonia Detection</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<h4 style='text-align: center; color: black;'>Upload a lung X-Ray image to know if the patient has bacterial or viral pneumonia</h4>",
        unsafe_allow_html=True,
    )
    st.markdown("#")
    uploaded_image = st.file_uploader("Upload an image to predict")

    if uploaded_image:
        st.image(uploaded_image)

        pred_button = st.button("Predict")
        if pred_button:
            prediction, confidence = predict(uploaded_image)
            print(prediction)
            if prediction == 0:
                st.subheader("The patient is not suffering from Pneumonia 😄🎉🎉")
                st.subheader(f"Confidence of model: {confidence*100:.2f}%")
                st.balloons()
            elif prediction == 1:
                st.subheader("The patient is suffering from Bacterial Pneumonia 😔")
                st.subheader(f"Confidence of model: {confidence*100:.2f}%")

            elif prediction == 2:
                st.subheader("The patient is suffering from Viral Pneumonia 😔")
                st.subheader(f"Confidence of model: {confidence*100:.2f}%")

            if prediction != 0:
                st.markdown("---")
                st.error(
                    "If you are a patient, consult with one of the following doctors immediately"
                )
                st.subheader("Specialists 👨‍⚕")
                st.write(
                    "Click on the specialist's name to find out the nearest specialist to you ..."
                )
                for s in ["Primary Care Doctor", "Lung Specialist"]:
                    doctor = f"- [{s}]({doctor_search(s)})"
                    st.markdown(doctor, unsafe_allow_html=True)
                st.markdown("---")


if __name__ == "__main__":
    model = PneumoniaModel()
    model.load_state_dict(torch.load(cfg.PNEUMONIA_MODEL, map_location="cpu"))
    model.eval()

    app()
