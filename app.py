import streamlit as st
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
from torchvision import transforms
from torchvision.models import mobilenet_v3_small


st.set_page_config(
    page_title="Leaf Nutrient Scanner",
    page_icon="🌱",
    layout="centered"
)

st.title("🌱 Leaf Nutrient Scanner")
st.write("Upload a leaf image to detect possible nutrient deficiency.")


CLASS_NAMES = ["-K", "-N", "-P", "FN"]

CLASS_DESCRIPTIONS = {
    "-K": "Potassium Deficiency",
    "-N": "Nitrogen Deficiency",
    "-P": "Phosphorus Deficiency",
    "FN": "Healthy / Fully Nutritional"
}

MODEL_PATH = Path(__file__).parent / "leaf_mobilenet_v2.pth"


@st.cache_resource
def load_model():
    model = mobilenet_v3_small(weights=None)

    model.classifier[3] = nn.Linear(
        model.classifier[3].in_features,
        len(CLASS_NAMES)
    )

    model.load_state_dict(
        torch.load(MODEL_PATH, map_location="cpu")
    )
    model.eval()

    return model


model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


uploaded_file = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Leaf", width="stretch")

    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        probabilities = torch.softmax(
            model(image_tensor),
            dim=1
        )

    predicted_class = probabilities.argmax(dim=1).item()
    prediction = CLASS_NAMES[predicted_class]
    confidence = probabilities[0, predicted_class].item() * 100

    st.subheader("Prediction")
    st.success(CLASS_DESCRIPTIONS[prediction])

    st.write(f"**Class:** {prediction}")
    st.write(f"**Confidence:** {confidence:.2f}%")

    st.subheader("Class Probabilities")

    for index, class_name in enumerate(CLASS_NAMES):
        probability = probabilities[0, index].item() * 100
        st.write(
            f"{CLASS_DESCRIPTIONS[class_name]}: "
            f"{probability:.2f}%"
        )
        st.progress(probability / 100)


st.divider()
st.caption(
    "This system uses MobileNetV3-Small trained on the Lettuce NPK dataset."
)
