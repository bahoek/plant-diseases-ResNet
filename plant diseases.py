import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.cuda.amp import autocast, GradScaler

import numpy as np
import os 
import matplotlib.pyplot as plt
import time

scaler = GradScaler()
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),        
    transforms.RandomHorizontalFlip(),   
    transforms.RandomRotation(10),       
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # 이미지넷 국룰 정규화값
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

data_dir = r"C:\data\archive (2)"

train_dataset = datasets.ImageFolder(
    root=os.path.join(data_dir, 'new plant diseases dataset(augmented)', 'New Plant Diseases Dataset(Augmented)',  'train'),
    transform=train_transform)
valid_dataset = datasets.ImageFolder(
    root=os.path.join(data_dir, 'new plant diseases dataset(augmented)', 'New Plant Diseases Dataset(Augmented)','valid'),
    transform=val_transform)


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels))
            
    def forward(self, x):
        identity = x  
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(identity) 
        out = F.relu(out)
        return out
    

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2) 
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512, num_classes)
        
    def _make_layer(self, block, out_channels, num_blocks, stride):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = self.avgpool(out) 
        out = torch.flatten(out, 1) 
        out = self.linear(out)
        return out

if __name__ == '__main__':
    train_loader = DataLoader(train_dataset, batch_size=64, num_workers=8, pin_memory=True,shuffle=True)  # 학습은 섞어서
    valid_loader = DataLoader(valid_dataset, batch_size=64, num_workers=8, pin_memory=True,shuffle=False) # 검증은 안 섞어도 됨
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    num_classes = len(train_dataset.classes)
    
    MODEL_DIR = './'
    if not os.path.exists(MODEL_DIR):
        os.mkdir(MODEL_DIR)
        
    history = {'loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    patience = 5
    counter = 0
    epochs = 30
    start_time = time.time()
    LR = 0.001
    
    model = ResNet(block=BasicBlock, num_blocks=[2, 2, 2, 2],num_classes=num_classes).to(device)    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=LR)

    
    for epoch in range(epochs):
        
        model.train()
        train_loss = 0
        for i,(images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()  # loss.backward() 대체
            scaler.step(optimizer)         # optimizer.step() 대체
            scaler.update()
            
            train_loss += loss.item() * images.size(0)
            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] Step [{i+1}/{len(train_loader)}] Loss: {loss.item():.4f}")
                
        train_loss /= len(train_loader.dataset)
        history['loss'].append(train_loss)

        
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()

        val_loss /= len(valid_loader.dataset)
        val_acc = correct / len(valid_loader.dataset)
        history['val_loss'].append(val_loss)

        print(f"Epoch {epoch+1}/{epochs} - loss: {train_loss:.4f} - val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}")

        # Checkpointer
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{MODEL_DIR}best_model.pth")
            counter = 0
            best_model_time = time.time() - start_time
            best_epoch_idx = epoch + 1
        else:
            counter += 1

        # Early Stopping
        if counter >= patience:
            print("Early Stopping!")
            break

    # 6. 결과 그래프 출력
    total_time = time.time() - start_time # 전체 걸린 시간

    print("-" * 50)
    print("🏁 Training Finished.")
    print(f"1. Total Training Time : {total_time // 60:.0f}m {total_time % 60:.0f}s")
    print(f"2. Time to Best Model  : {best_model_time // 60:.0f}m {best_model_time % 60:.0f}s (at Epoch {best_epoch_idx})")
    print("-" * 50)
    plt.plot(history['val_loss'], marker='.', c="red", label='Testset_loss')
    plt.plot(history['loss'], marker='.', c="blue", label='Trainset_loss')
    plt.legend(loc='upper right')
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()