import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os


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
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = r"C:\data\archive (2)"
    model_path = './model/best_model.pth'
    
    # 2. 테스트 데이터셋 준비
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print("📁 테스트 데이터를 불러오는 중...")
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'new plant diseases dataset(augmented)', 'New Plant Diseases Dataset(Augmented)',  'valid'), transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4,pin_memory=True)
    
    
    class_names = test_dataset.classes
    num_classes = len(class_names)
    print(f"📊 감지된 클래스 개수: {num_classes}개")

    
    print("🧠 모델을 로드하는 중...")
    model = ResNet(block=BasicBlock, num_blocks=[2, 2, 2, 2], num_classes=num_classes).to(device)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        model.eval() # ★ 평가 모드 (필수!)
        print("✅ 학습된 모델 로드 완료!")
    else:
        print(f"❌ 모델 파일이 없습니다: {model_path}")
        exit()

    # 4. 전체 정확도 평가
    print("🚀 정확도 측정 시작...")
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"\n🏆 최종 테스트 정확도: {accuracy:.2f}%")
    
# 결과 눈으로 확인
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    images = images.to(device)
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

    # 시각화 함수
    def imshow(img, title):
        img = img.cpu().numpy().transpose((1, 2, 0))
        img = np.array([0.229, 0.224, 0.225]) * img + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        plt.imshow(img)
        plt.title(title, fontsize=10)
        plt.axis('off')

    plt.figure(figsize=(12, 8))
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        color = 'blue' if predicted[i] == labels[i] else 'red'
        title = f"Pred: {class_names[predicted[i]]}\nActual: {class_names[labels[i]]}"
        imshow(images[i], title)
    plt.show()