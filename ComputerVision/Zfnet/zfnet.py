import torch
import numpy as np
import matplotlib.pyplot as plt
import torchsummary
import torchvision
import os

current_file = os.path.splitext(os.path.basename(__file__))[0]
print("Current file name:", current_file)


class ZFNet(torch.nn.Module):
    def __init__(self, num_classes=10):
        super(ZFNet, self).__init__()

        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(1, 96, kernel_size=7, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            torch.nn.Conv2d(96, 256, kernel_size=5, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            torch.nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2)
        )
        
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(256 * 5 * 5, 512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(),
            torch.nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class MnistDataset(torch.utils.data.Dataset):
    def __init__(self, root, train=True):
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.Grayscale(num_output_channels=1),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.dataset = torchvision.datasets.MNIST(
            root=root, train=train, transform=self.transform, download=True)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]


def train(train_loader, test_loader):
    # Update num_classes based on your dataset
    model = ZFNet(num_classes=10)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    num_epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    torchsummary.summary(model, input_size=(1, 224, 224))
    
    for epoch in range(num_epochs):
        train_loss = 0.0
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if (batch_idx + 1) % 100 == 0:
                print('Epoch: {}, Batch: {}, Avg. Loss: {:.4f}'.format(
                    epoch + 1, batch_idx + 1, train_loss / 100))
                train_loss = 0.0

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        accuracy = 100.0 * correct / total
        print('Accuracy on the test set: {:.2f}%'.format(accuracy))

        print('Model saved...')
        torch.save(model.state_dict(), 'models/' + current_file+'_model.pth')

    print('Model Loaded...')
    model = ZFNet(num_classes=10)
    model.load_state_dict(torch.load('models/' + current_file+'_model.pth'))


if __name__ == '__main__':
    train_dataset = MnistDataset(root='./data', train=True)
    test_dataset = MnistDataset(root='./data', train=False)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=16, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=16, shuffle=False)

    train(train_loader, test_loader)
