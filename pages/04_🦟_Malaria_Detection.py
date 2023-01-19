import streamlit as st

import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import config as cfg
from config import doctor_search
import torch


class MosquitoNet(nn.Module):
    def __init__(self):
        super(MosquitoNet, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc1 = nn.Linear(64 * 15 * 15, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 2)
        self.drop = nn.Dropout2d(0.2)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)  # flatten out a input for Dense Layer
        out = self.fc1(out)
        out = F.relu(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.drop(out)
        out = self.fc3(out)

        return out


def malaria_app():
    st.set_page_config(
        page_title="Healthify - Malaria Detection",
        page_icon="🏥",
    )
    st.markdown(
        f"<h1 style='text-align: center; color: black;'>Malaria Detection</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<h4 style='text-align: center; color: black;'>Upload a microscopic cell image to know if the patient has malaria or not</h4>",
        unsafe_allow_html=True,
    )
    malaria_model = MosquitoNet()
    malaria_model.load_state_dict(
        torch.load(cfg.MALARIA_MODEL, map_location=torch.device("cpu"))
    )

    malaria_model.eval()

    uploaded_image = st.file_uploader("Upload a cell image")

    if uploaded_image:
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption="Uploaded image")

        predict = st.button("Predict")

        if predict:
            prediction_transform = transforms.Compose(
                [
                    transforms.Resize((120, 120)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            )

            image = prediction_transform(image).unsqueeze(0)

            output = malaria_model(image)
            predicted = torch.max(output, 1)[1]
            if predicted == 0:
                st.subheader("The patient is infected with malaria 😔")
                st.markdown("---")
                st.error(
                    "If you are a patient, consult with one of the following doctors immediately"
                )
                st.subheader("Specialists 👨‍⚕")
                st.write(
                    "Click on the specialists to get the specialists nearest to your location 📍"
                )

                pcp = doctor_search("Primary Care Provider")
                infec = doctor_search("Infectious Disease Doctor")
                st.markdown(f"- [Primary Care Doctor]({pcp}) 👨‍⚕")
                st.markdown(f"- [Infectious Disease Doctor]({infec}) 👨‍⚕")
                st.markdown("---")
            else:
                st.subheader(
                    "The patient with the given cell image is un-infected with malaria 😄"
                )
                st.balloons()


if __name__ == "__main__":
    malaria_app()
