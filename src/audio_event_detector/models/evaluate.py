import torch
from sklearn.metrics import accuracy_score, confusion_matrix


def evaluate(model, dataloader, device):

    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():

        for x, y in dataloader:

            x = x.to(device)

            logits = model(x)

            preds = torch.argmax(logits, dim=1).cpu().numpy()

            y_true.extend(y.numpy())
            y_pred.extend(preds)

    acc = accuracy_score(y_true, y_pred)

    return acc, confusion_matrix(y_true, y_pred)
