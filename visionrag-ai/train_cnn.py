"""Train on data/train/<class>/*.png and data/val/<class>/*.png."""
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from cnn import DocumentCNN, CLASSES

def main():
    transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
    train = datasets.ImageFolder("data/train", transform=transform)
    val = datasets.ImageFolder("data/val", transform=transform)
    if train.classes != CLASSES or val.classes != CLASSES:
        raise ValueError(f"Expected class folders: {CLASSES}")
    train_loader = DataLoader(train, batch_size=16, shuffle=True)
    val_loader = DataLoader(val, batch_size=16)
    model = DocumentCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(10):
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                pred = model(images).argmax(1)
                correct += (pred == labels).sum().item()
                total += len(labels)
        print(f"Epoch {epoch+1}: validation accuracy={correct / max(total,1):.3f}")
    Path("models").mkdir(exist_ok=True)
    torch.save(model.state_dict(), "models/document_cnn.pt")

if __name__ == "__main__":
    main()
