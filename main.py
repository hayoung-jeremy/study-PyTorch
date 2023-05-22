import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# 공개 데이터 셋에서 학습 데이터를 내려받음
training_data = datasets.FashionMNIST(
    root="data", train=True, download=True, transform=ToTensor()
)

# 공개 데이터셋에서 테스트 데이터 다운로드
test_data = datasets.FashionMNIST(
    root="data", train=False, download=True, transform=ToTensor()
)

# Dataset을 DataLoader의 인자로 전달하여, 순회 가능한 객체로 감싼다
batch_size = 64  # dataloader 객체의 각 요소는 64개의 특징(feature)과 정답(label)을 묶음(batch)으로 반환한다.

train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shpae of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y : {y.shape} {y.dtype}")
    break

# 학습에 사용할 CPU, GPU, 또는 MPS 장치 불러오기
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using {device} device")


# define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork().to(device)
print(model)
