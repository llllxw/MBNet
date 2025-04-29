import torch

def training_classing(model, data_loader, optimizer, criterion, device):
    model.train()
    total_loss = []
    for _, data in enumerate(data_loader):
        data = data.to(device)
        y = data.y.view(-1, 1).to(device)  # (b, 1)
        optimizer.zero_grad()
        output, _ = model(data)
        loss = criterion(output, y)
        total_loss.append(loss.item())
        loss.backward()
        optimizer.step()
    return sum(total_loss) / len(total_loss)

def evaluate_test_scores(model, data_loader, criterion, device):
    model.eval()
    total_loss = []
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            y = data.y.view(-1, 1).to(device)
            y_pred, _ = model(data)
            total_loss.append(criterion(y_pred, y).item())
            y_pred = torch.sigmoid(y_pred)
            y_true = y.squeeze()
            targets, pred_scores = y_true, y_pred
            total_preds = torch.cat([total_preds, pred_scores.cpu()], dim=0)
            total_labels = torch.cat([total_labels, targets.cpu()], dim=0)
            
    return sum(total_loss) / len(total_loss), total_labels.numpy().flatten(), total_preds.numpy().flatten()
