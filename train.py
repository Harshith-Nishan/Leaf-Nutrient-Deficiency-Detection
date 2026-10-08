import torch
import torch.nn as nn
from collections import Counter
from pathlib import Path

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


PROJECT_DIR = Path(__file__).parent
DATASET_PATH = PROJECT_DIR / "dataset" / "npk"
MODEL_PATH = PROJECT_DIR / "leaf_mobilenet_v2.pth"

BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.001


base_transform = transforms.ToTensor()

dataset = datasets.ImageFolder(
    root=DATASET_PATH,
    transform=base_transform
)

print("Classes:", dataset.classes)
print("Class mapping:", dataset.class_to_idx)
print("Total images:", len(dataset))


labels = [label for _, label in dataset]
counts = Counter(labels)

print("\nImages per class:")
for class_name, class_index in dataset.class_to_idx.items():
    print(f"{class_name}: {counts[class_index]}")


indices = list(range(len(dataset)))

train_indices, val_indices = train_test_split(
    indices,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

print("\nTraining images:", len(train_indices))
print("Validation images:", len(val_indices))


train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


train_dataset = Subset(
    datasets.ImageFolder(
        root=DATASET_PATH,
        transform=train_transform
    ),
    train_indices
)

val_dataset = Subset(
    datasets.ImageFolder(
        root=DATASET_PATH,
        transform=val_transform
    ),
    val_indices
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


print("\nTraining batches:", len(train_loader))
print("Validation batches:", len(val_loader))


weights = MobileNet_V3_Small_Weights.DEFAULT

model = mobilenet_v3_small(weights=weights)

for parameter in model.features.parameters():
    parameter.requires_grad = False

model.classifier[3] = nn.Linear(
    model.classifier[3].in_features,
    4
)


criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.classifier[3].parameters(),
    lr=LEARNING_RATE
)


print("\nStarting training...")

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    average_loss = running_loss / len(train_loader)

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Loss: {average_loss:.4f}"
    )


model.eval()

correct = 0
total = 0

with torch.no_grad():
    for images, labels in val_loader:
        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        total += labels.size(0)
        correct += (predictions == labels).sum().item()

accuracy = 100 * correct / total

print(f"\nValidation Accuracy: {accuracy:.2f}%")


all_predictions = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())


print("\nConfusion Matrix:")
print(confusion_matrix(all_labels, all_predictions))

print("\nClassification Report:")
print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=["-K", "-N", "-P", "FN"],
        zero_division=0
    )
)


torch.save(model.state_dict(), MODEL_PATH)

print(f"\nModel saved to: {MODEL_PATH}")
