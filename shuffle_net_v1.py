import torch
import numpy as np
import matplotlib.pyplot as plt
import torchsummary
import torchvision
import os

current_file = os.path.splitext(os.path.basename(__file__))[0]
print("Current file name:", current_file)


class ShuffleBlock(torch.nn.Module):
    def __init__(self, groups):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        N, C, H, W = x.size()
        g = self.groups
        return x.view(N, g, C//g, H, W).permute(0, 2, 1, 3, 4).reshape(N, C, H, W)


class Bottleneck(torch.nn.Module):
    def __init__(self, in_planes, out_planes, stride, groups):
        super(Bottleneck, self).__init__()
        self.stride = stride
        mid_planes = int(out_planes/4)
        g = 1 if in_planes == 24 else groups
        self.conv1 = torch.nn.Conv2d(in_planes, mid_planes,
                                     kernel_size=1, groups=g, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(mid_planes)
        self.shuffle1 = ShuffleBlock(groups=g)
        self.conv2 = torch.nn.Conv2d(mid_planes, mid_planes, kernel_size=3,
                                     stride=stride, padding=1, groups=mid_planes, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(mid_planes)
        self.conv3 = torch.nn.Conv2d(mid_planes, out_planes,
                                     kernel_size=1, groups=groups, bias=False)
        self.bn3 = torch.nn.BatchNorm2d(out_planes)
        self.shortcut = torch.nn.Sequential()
        if stride == 2:
            self.shortcut = torch.nn.Sequential(
                torch.nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, x):
        out = torch.nn.functional.relu(self.bn1(self.conv1(x)))
        out = self.shuffle1(out)
        out = torch.nn.functional.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        res = self.shortcut(x)
        out = torch.nn.functional.relu(
            torch.cat([out, res], 1)) if self.stride == 2 else torch.nn.functional.relu(out+res)
        return out


class ShuffleNet(torch.nn.Module):
    def __init__(self, cfg):
        super(ShuffleNet, self).__init__()
        out_planes = cfg['out_planes']
        num_blocks = cfg['num_blocks']
        groups = cfg['groups']
        self.conv1 = torch.nn.Conv2d(1, 24, kernel_size=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(24)
        self.in_planes = 24
        self.layer1 = self._make_layer(out_planes[0], num_blocks[0], groups)
        self.layer2 = self._make_layer(out_planes[1], num_blocks[1], groups)
        self.layer3 = self._make_layer(out_planes[2], num_blocks[2], groups)
        self.linear = torch.nn.Linear(out_planes[2], 10)

    def _make_layer(self, out_planes, num_blocks, groups):
        layers = []
        for i in range(num_blocks):
            stride = 2 if i == 0 else 1
            cat_planes = self.in_planes if i == 0 else 0
            layers.append(Bottleneck(self.in_planes, out_planes -
                          cat_planes, stride=stride, groups=groups))
            self.in_planes = out_planes
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        out = torch.nn.functional.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = torch.nn.functional.avg_pool2d(out, 20)
        out = out.view(out.size(0), -1)
        out = torch.flatten(out,1)
        out = self.linear(out)
        return out


def ShuffleNetG2():
    cfg = {'out_planes': [200, 400, 800],
           'num_blocks': [4, 8, 4],
           'groups': 2
           }
    return ShuffleNet(cfg)


def ShuffleNetG3():
    cfg = {'out_planes': [240, 480, 960],
           'num_blocks': [4, 8, 4],
           'groups': 3
           }
    return ShuffleNet(cfg)


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
    model = ShuffleNetG2()
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
    model = ShuffleNetG2()
    model.load_state_dict(torch.load('models/' + current_file+'_model.pth'))


if __name__ == '__main__':
    train_dataset = MnistDataset(root='./data', train=True)
    test_dataset = MnistDataset(root='./data', train=False)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=16, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=16, shuffle=False)

    train(train_loader, test_loader)
