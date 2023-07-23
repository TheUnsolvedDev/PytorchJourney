import torch
import numpy as np
import matplotlib.pyplot as plt
import torchsummary
import torchvision
import os

current_file = os.path.splitext(os.path.basename(__file__))[0]
print("Current file name:", current_file)


def channel_shuffle(x, groups=2):
    bat_size, channels, w, h = x.shape
    group_c = channels // groups
    x = x.view(bat_size, groups, group_c, w, h)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(bat_size, -1, w, h)
    return x

# used in the block


def conv_1x1_bn(in_c, out_c, stride=1):
    return torch.nn.Sequential(
        torch.nn.Conv2d(in_c, out_c, 1, stride, 0, bias=False),
        torch.nn.BatchNorm2d(out_c),
        torch.nn.ReLU(True)
    )


def conv_bn(in_c, out_c, stride=2):
    return torch.nn.Sequential(
        torch.nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False),
        torch.nn.BatchNorm2d(out_c),
        torch.nn.ReLU(True)
    )


class ShuffleBlock(torch.nn.Module):
    def __init__(self, in_c, out_c, downsample=False):
        super(ShuffleBlock, self).__init__()
        self.downsample = downsample
        half_c = out_c // 2
        if downsample:
            self.branch1 = torch.nn.Sequential(
                torch.nn.Conv2d(in_c, in_c, 3, 2, 1, groups=in_c, bias=False),
                torch.nn.BatchNorm2d(in_c),
                torch.nn.Conv2d(in_c, half_c, 1, 1, 0, bias=False),
                torch.nn.BatchNorm2d(half_c),
                torch.nn.ReLU(True)
            )

            self.branch2 = torch.nn.Sequential(
                torch.nn.Conv2d(in_c, half_c, 1, 1, 0, bias=False),
                torch.nn.BatchNorm2d(half_c),
                torch.nn.ReLU(True),
                torch.nn.Conv2d(half_c, half_c, 3, 2, 1, groups=half_c, bias=False),
                torch.nn.BatchNorm2d(half_c),
                torch.nn.Conv2d(half_c, half_c, 1, 1, 0, bias=False),
                torch.nn.BatchNorm2d(half_c),
                torch.nn.ReLU(True)
            )
        else:
            assert in_c == out_c

            self.branch2 = torch.nn.Sequential(
                torch.nn.Conv2d(half_c, half_c, 1, 1, 0, bias=False),
                torch.nn.BatchNorm2d(half_c),
                torch.nn.ReLU(True),
                # 3*3 dw conv, stride = 1
                torch.nn.Conv2d(half_c, half_c, 3, 1, 1, groups=half_c, bias=False),
                torch.nn.BatchNorm2d(half_c),
                # 1*1 pw conv
                torch.nn.Conv2d(half_c, half_c, 1, 1, 0, bias=False),
                torch.nn.BatchNorm2d(half_c),
                torch.nn.ReLU(True)
            )

    def forward(self, x):
        out = None
        if self.downsample:
            out = torch.cat((self.branch1(x), self.branch2(x)), 1)
        else:
            channels = x.shape[1]
            c = channels // 2
            x1 = x[:, :c, :, :]
            x2 = x[:, c:, :, :]
            out = torch.cat((x1, self.branch2(x2)), 1)
        return channel_shuffle(out, 2)


class ShuffleNet2(torch.nn.Module):
    def __init__(self, num_classes=10, input_size=224, net_type=1):
        super(ShuffleNet2, self).__init__()
        assert input_size % 32 == 0 

        self.stage_repeat_num = [4, 8, 4]
        if net_type == 0.5:
            self.out_channels = [3, 24, 48, 96, 192, 1024]
        elif net_type == 1:
            self.out_channels = [3, 24, 116, 232, 464, 1024]
        elif net_type == 1.5:
            self.out_channels = [3, 24, 176, 352, 704, 1024]
        elif net_type == 2:
            self.out_channels = [3, 24, 244, 488, 976, 2948]
        else:
            print("the type is error, you should choose 0.5, 1, 1.5 or 2")

        self.conv1 = torch.nn.Conv2d(1, self.out_channels[1], 3, 2, 1)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        in_c = self.out_channels[1]

        self.stages = []
        for stage_idx in range(len(self.stage_repeat_num)):
            out_c = self.out_channels[2+stage_idx]
            repeat_num = self.stage_repeat_num[stage_idx]
            for i in range(repeat_num):
                if i == 0:
                    self.stages.append(ShuffleBlock(
                        in_c, out_c, downsample=True))
                else:
                    self.stages.append(ShuffleBlock(
                        in_c, in_c, downsample=False))
                in_c = out_c
        self.stages = torch.nn.Sequential(*self.stages)

        in_c = self.out_channels[-2]
        out_c = self.out_channels[-1]
        self.conv5 = conv_1x1_bn(in_c, out_c, 1)
        self.g_avg_pool = torch.nn.AvgPool2d(kernel_size=(
            int)(input_size/32))  

        self.fc = torch.nn.Linear(out_c, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stages(x)
        x = self.conv5(x)
        x = self.g_avg_pool(x)
        x = x.view(-1, self.out_channels[-1])
        x = self.fc(x)
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
    model = ShuffleNet2()
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
    model = ShuffleNet2()
    model.load_state_dict(torch.load('models/' + current_file+'_model.pth'))


if __name__ == '__main__':
    train_dataset = MnistDataset(root='./data', train=True)
    test_dataset = MnistDataset(root='./data', train=False)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=64, shuffle=False)

    train(train_loader, test_loader)
