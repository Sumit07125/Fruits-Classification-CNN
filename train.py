import os
import time
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights

def get_device():
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        return torch.device('xpu')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def sync_device(device):
    if device.type == 'xpu' and hasattr(torch, 'xpu'):
        torch.xpu.synchronize()
    elif device.type == 'cuda':
        torch.cuda.synchronize()
device = get_device()
print('Using device:', device)
data_dir = 'D:\\CODE\\gpu\\New folder\\Fruits Classification'
train_transform = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.RandomRotation(15), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
valid_transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
train_data = datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=train_transform)
valid_data = datasets.ImageFolder(root=os.path.join(data_dir, 'valid'), transform=valid_transform)
test_data = datasets.ImageFolder(root=os.path.join(data_dir, 'test'), transform=valid_transform)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=0)
valid_loader = DataLoader(valid_data, batch_size=32, shuffle=False, num_workers=0)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=0)
print('Classes:', train_data.classes)
print('Number of classes:', len(train_data.classes))
print('Train samples:', len(train_data))
print('Valid samples:', len(valid_data))
print('Test samples:', len(test_data))
print('Train batches:', len(train_loader))
print('Valid batches:', len(valid_loader))
print('Test batches:', len(test_loader))
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
model.fc = nn.Linear(model.fc.in_features, len(train_data.classes))
model = model.to(device)
print(model)
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (images, labels) in enumerate(loader, start=1):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        print(f'Train Batch {batch_idx}/{len(loader)} | Seen {total}/{len(loader.dataset)} | Loss {loss.item():.4f}', end='\r')
    print()
    return (total_loss / total, 100.0 * correct / total)

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader, start=1):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            print(f'Valid Batch {batch_idx}/{len(loader)} | Seen {total}/{len(loader.dataset)} | Loss {loss.item():.4f}', end='\r')
    print()
    return (total_loss / total, 100.0 * correct / total)
for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():
    param.requires_grad = True
optimizer = torch.optim.AdamW(model.fc.parameters(), lr=0.001, weight_decay=0.0001)
best_val_acc = 0.0
os.makedirs('models', exist_ok=True)
save_path = os.path.join('models', 'fruit_resnet50_best.pt')
total_start = time.perf_counter()
print('\nStage 1: training classifier head')
stage1_epochs = 5
for epoch in range(1, stage1_epochs + 1):
    print(f'\nEpoch {epoch}/{stage1_epochs}')
    epoch_start = time.perf_counter()
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    sync_device(device)
    val_loss, val_acc = evaluate(model, valid_loader, criterion, device)
    sync_device(device)
    epoch_time = time.perf_counter() - epoch_start
    print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
    print(f'Valid Loss: {val_loss:.4f} | Valid Acc: {val_acc:.2f}%')
    print(f'Epoch Time: {int(epoch_time // 60)}m {epoch_time % 60:.1f}s')
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({'model_state_dict': model.state_dict(), 'class_names': train_data.classes}, save_path)
        print(f'Saved best model to {save_path}')
print('\nStage 2: fine-tuning all layers')
for param in model.parameters():
    param.requires_grad = True
optimizer = torch.optim.AdamW([{'params': model.layer4.parameters(), 'lr': 1e-05}, {'params': model.layer3.parameters(), 'lr': 5e-06}, {'params': model.layer2.parameters(), 'lr': 5e-06}, {'params': model.layer1.parameters(), 'lr': 5e-06}, {'params': model.conv1.parameters(), 'lr': 5e-06}, {'params': model.bn1.parameters(), 'lr': 5e-06}, {'params': model.fc.parameters(), 'lr': 0.0001}], weight_decay=0.0001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
stage2_epochs = 20
for epoch in range(1, stage2_epochs + 1):
    print(f'\nEpoch {epoch}/{stage2_epochs}')
    epoch_start = time.perf_counter()
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    sync_device(device)
    val_loss, val_acc = evaluate(model, valid_loader, criterion, device)
    sync_device(device)
    scheduler.step(val_acc)
    epoch_time = time.perf_counter() - epoch_start
    print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
    print(f'Valid Loss: {val_loss:.4f} | Valid Acc: {val_acc:.2f}%')
    print(f'Epoch Time: {int(epoch_time // 60)}m {epoch_time % 60:.1f}s')
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({'model_state_dict': model.state_dict(), 'class_names': train_data.classes}, save_path)
        print(f'Saved best model to {save_path}')
total_time = time.perf_counter() - total_start
print(f'\nTotal Training Time: {int(total_time // 60)}m {total_time % 60:.1f}s')
checkpoint = torch.load(save_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
test_correct = 0
test_total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()
test_acc = 100.0 * test_correct / test_total
print(f'Test Accuracy: {test_acc:.2f}%')
print('Done.')
