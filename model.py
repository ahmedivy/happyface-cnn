import sys
import torch
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms


class HappyCNN(nn.Module):
    def __init__(self, input_channels=3):
        super(HappyCNN, self).__init__()
        self.conv = nn.Conv2d(input_channels, 32, kernel_size=7, stride=1, padding=3)
        self.bn = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * 32 * 32, 1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = F.sigmoid(x)
        return x


def predict(img_path):
    # Load model
    model = HappyCNN()
    model.load_state_dict(torch.load('model/happy-cnn.pth'))
    model.eval()

    # Load image
    img = Image.open(img_path)
    img = img.resize((64, 64))
    img = transforms.ToTensor()(img)
    img = img.unsqueeze(0)

    # Predict
    with torch.no_grad():
        pred = model(img)
        pred = pred.squeeze()
        pred = pred.item()
        if pred > 0.5:
            print('Happy')
        else:
            print('Unhappy')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python model.py <image_path>')
        exit(1)

    predict(sys.argv[1])
