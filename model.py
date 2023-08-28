import sys
import h5py
import torch
import argparse
import numpy as np
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


def load_dataset():
    train_dataset = h5py.File('datasets/train_happy.h5', "r")
    train_x = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_y = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File('datasets/test_happy.h5', "r")
    test_x = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_y = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_y = train_y.reshape((1, train_y.shape[0]))
    test_y = test_y.reshape((1, test_y.shape[0]))

    return train_x, train_y, test_x, test_y, classes


def train(epochs = 20, batch_size = 32):
    # Load dataset
    train_x, train_y, test_x, test_y, _ = load_dataset()

    # Normalize
    train_x = train_x / 255.0
    test_x = test_x / 255.0

    # Convert to tensor
    train_x = torch.from_numpy(train_x).float()
    train_y = torch.from_numpy(train_y).float()
    test_x = torch.from_numpy(test_x).float()
    test_y = torch.from_numpy(test_y).float()

    # Create model
    model = HappyCNN()

    # Loss function
    criterion = nn.BCELoss()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train
    for epoch in range(epochs):
        for i in range(0, len(train_x), batch_size):
            batch_x = train_x[i:i+batch_size]
            batch_y = train_y[i:i+batch_size]

            optimizer.zero_grad()
            y_pred = model(batch_x)
            loss = criterion(y_pred, batch_y)
            loss.backward()
            optimizer.step()

        print(f'Epoch: {epoch+1}, Loss: {loss.item():.3f}')


    # Save model
    torch.save(model.state_dict(), 'model/happy-cnn.pth')


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
    
    parser = argparse.ArgumentParser(description='Happy CNN')
    parser.add_argument('--train', action='store_true', help='Train model')
    parser.add_argument('--predict', type=str, help='Predict image')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    args = parser.parse_args()

    if args.train:
        train(args.epochs, args.batch_size)
    elif args.predict:
        predict(args.predict)
    else:
        parser.print_help()
        sys.exit(1)
