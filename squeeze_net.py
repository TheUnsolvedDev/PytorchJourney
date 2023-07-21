import torch
import numpy as np
import matplotlib.pyplot as plt
import torchsummary
import torchvision
import os

current_file = os.path.splitext(os.path.basename(__file__))[0]
print("Current file name:", current_file)


class FireModule(torch.nn.Module):
    def __init__(self, in_channels, s1x1, e1x1, e3x3):
        super(FireModule, self).__init__()
        self.squeeze = torch.nn.Conv2d(
            in_channels=in_channels, out_channels=s1x1, kernel_size=1, stride=1)
        self.expand1x1 = torch.nn.Conv2d(
            in_channels=s1x1, out_channels=e1x1, kernel_size=1)
        self.expand3x3 = torch.nn.Conv2d(
            in_channels=s1x1, out_channels=e3x3, kernel_size=3, padding=1)

    def forward(self, x):
        x = torch.nn.functional.relu(self.squeeze(x))
        x1 = self.expand1x1(x)
        x2 = self.expand3x3(x)
        x = torch.nn.functional.relu(torch.cat((x1, x2), dim=1))
        return x


class SqueezeNet(torch.nn.Module):
    def __init__(self, out_channels=10):
        super(SqueezeNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels=1, out_channels=96, kernel_size=7, stride=2)
        self.max_pool1 = torch.nn.MaxPool2d(kernel_size=3, stride=2)

        self.fire2 = FireModule(in_channels=96, s1x1=16, e1x1=64, e3x3=64)
        self.fire3 = FireModule(in_channels=128, s1x1=16, e1x1=64, e3x3=64)
        self.fire4 = FireModule(in_channels=128, s1x1=32, e1x1=128, e3x3=128)
        self.max_pool2 = torch.nn.MaxPool2d(kernel_size=3, stride=2)

        self.fire5 = FireModule(in_channels=256, s1x1=32, e1x1=128, e3x3=128)
        self.fire6 = FireModule(in_channels=256, s1x1=48, e1x1=192, e3x3=192)
        self.fire7 = FireModule(in_channels=384, s1x1=48, e1x1=192, e3x3=192)
        self.fire8 = FireModule(in_channels=384, s1x1=64, e1x1=256, e3x3=256)
        self.max_pool3 = torch.nn.MaxPool2d(kernel_size=3, stride=2)

        self.fire9 = FireModule(in_channels=512, s1x1=64, e1x1=256, e3x3=256)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.conv10 = torch.nn.Conv2d(
            in_channels=512, out_channels=out_channels, kernel_size=1, stride=1)
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=12, stride=1)
        # We don't have FC Layers, inspired by NiN architecture.

    def forward(self, x):
        x = self.max_pool1(self.conv1(x))
        x = self.max_pool2(self.fire4(self.fire3(self.fire2(x))))
        x = self.max_pool3(self.fire8(self.fire7(self.fire6(self.fire5(x)))))
        x = self.avg_pool(self.conv10(self.fire9(x)))
        return torch.flatten(x, start_dim=1)


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
    model = SqueezeNet()
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
    model = SqueezeNet()
    model.load_state_dict(torch.load('models/' + current_file+'_model.pth'))


if __name__ == '__main__':
    train_dataset = MnistDataset(root='./data', train=True)
    test_dataset = MnistDataset(root='./data', train=False)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=64, shuffle=False)

    train(train_loader, test_loader)