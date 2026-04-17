import torch
from torch.utils.data import DataLoader, random_split

from audio_event_detector.dataset.esc50_torch import ESC50Torch
from audio_event_detector.models.cnn import AudioCNN
from audio_event_detector.models.evaluate import evaluate

TARGET_CLASSES = [
    "dog",
    "cat",
    "crying_baby",
    "glass_breaking",
    "siren"
]

def train(root="data/raw/esc50", epochs=30, batch_size=32):

    dataset = ESC50Torch(root, target_classes=TARGET_CLASSES)
    
    print("dataset size:", len(dataset))
    print("classes:", dataset.classes)


    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    device = torch.device("cpu")

    model = AudioCNN(num_classes=len(dataset.classes)).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    best_val_acc = 0

    for epoch in range(epochs):

        model.train()

        total_loss = 0

        for x, y in train_loader:

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            output = model(x)

            loss = criterion(output, y)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()
        acc, _ = evaluate(model, val_loader, device)
        
        if acc > best_val_acc:
            best_val_acc = acc
            torch.save(model.state_dict(), "model_best.pt")


        print(f"epoch {epoch+1} loss {total_loss:.3f} val_acc{acc:.3f}")

    return model
