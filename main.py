
import os
from PIL import Image
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Dataset
import glob
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import resnet34
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 하이퍼파라미터 설정
batch_size = 16
num_epochs = 50
learning_rate = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RoadDataset(Dataset):
    def __init__(self, image_paths, label_paths, transform=None):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index])
        label = Image.open(self.label_paths[index])

        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        return image, label

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

train_dir = "C:/Users/verac/PycharmProjects/loac_detect/data/train/"
label_dir = "C:/Users/verac/PycharmProjects/loac_detect/data/label/"

image_paths = sorted(glob.glob(os.path.join(train_dir, '*.png')))
label_paths = sorted(glob.glob(os.path.join(label_dir, '*.png')))

dataset = RoadDataset(image_paths, label_paths, transform)

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 데이터 로더 시각화 부분
# images, labels = next(iter(dataloader))
#
# fig, axes = plt.subplots(len(images), 2, figsize=(10, 10))
#
# for idx, (image, label) in enumerate(zip(images, labels)):
#     # 입력 이미지 출력
#     axes[idx, 0].imshow(image.permute(1, 2, 0))
#     axes[idx, 0].axis('off')
#     axes[idx, 0].set_title('Input Image')
#
#     # 도로 레이블 출력
#     axes[idx, 1].imshow(label.squeeze(), cmap='gray')
#     axes[idx, 1].axis('off')
#     axes[idx, 1].set_title('Road Label')
#
# plt.tight_layout()
# plt.show()

# 도로 픽셀을 학습
class RoadPixelModel(nn.Module):
    def __init__(self):
        super(RoadPixelModel, self).__init__()
        # 모델 구조
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.sigmoid(out)
        return out

# 도로 픽셀 모델 인스턴스 생성
model = RoadPixelModel()

# 손실 함수
criterion = nn.BCELoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

# 학습 루프
num_epochs = 50
for epoch in range(num_epochs):
    for images, labels in dataloader:
        # 이미지와 레이블 배치를 모델에 입력
        outputs = model(images)

        # 손실
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

torch.save(model.state_dict(), 'road_pixel_model.pth')

model = RoadPixelModel()
model.load_state_dict(torch.load('road_pixel_model.pth'))
model.eval()

# 입력 이미지 로드 및 전처리
image_path = "C:/Users/verac/PycharmProjects/loac_detect/data/test/00003608_00000_20480_1024_1024.png"
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])
input_image = Image.open(image_path)
input_tensor = transform(input_image)
input_batch = input_tensor.unsqueeze(0)

# 도로 픽셀 예측
with torch.no_grad():
    output = model(input_batch)

# 결과 시각화
predicted_image = output.squeeze().numpy()
predicted_image[predicted_image < 0.5] = 0
predicted_image[predicted_image >= 0.5] = 1

# 입력 이미지 및 예측 결과 출력
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# 입력 이미지 출력
axes[0].imshow(input_image)
axes[0].axis('off')
axes[0].set_title('Input Image')

# 도로 픽셀 예측 결과 출력
axes[1].imshow(predicted_image, cmap='gray')
axes[1].axis('off')
axes[1].set_title('Predicted Road Pixels')

plt.tight_layout()
plt.show()
