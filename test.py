import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
from torchvision import transforms
from torchvision.models import mobilenet_v3_small


PROJECT_DIR = Path(__file__).parent
MODEL_PATH = PROJECT_DIR / "leaf_mobilenet_v2.pth"
IMAGE_PATH = PROJECT_DIR / "test_images" / "k_9.png"

CLASS_NAMES = ["-K", "-N", "-P", "FN"]


model = mobilenet_v3_small(weights=None)

model.classifier[3] = nn.Linear(
    model.classifier[3].in_features,
    len(CLASS_NAMES)
)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location="cpu")
)
model.eval()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


if not IMAGE_PATH.exists():
    raise FileNotFoundError(
        f"Test image not found: {IMAGE_PATH}"
    )


image = Image.open(IMAGE_PATH).convert("RGB")
image_tensor = transform(image).unsqueeze(0)


with torch.no_grad():
    probabilities = torch.softmax(
        model(image_tensor),
        dim=1
    )

predicted_class = probabilities.argmax(dim=1).item()
prediction = CLASS_NAMES[predicted_class]
confidence = probabilities[0, predicted_class].item() * 100


print(f"Prediction: {prediction}")
print(f"Confidence: {confidence:.2f}%")
