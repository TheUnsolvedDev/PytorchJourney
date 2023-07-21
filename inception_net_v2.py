import torch
import numpy as np
import matplotlib.pyplot as plt
import torchsummary
import torchvision
import os

current_file = os.path.splitext(os.path.basename(__file__))[0]
print("Current file name:", current_file)


class GridReduction(torch.nn.Module):
    def __init__(self, in_fts, out_fts):
        super(GridReduction, self).__init__()
        self.branch1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_fts, out_channels=out_fts,
                            kernel_size=(3, 3), stride=(2, 2))
        )

        self.branch2 = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        )

    def forward(self, input_img):
        o1 = self.branch1(input_img)
        o2 = self.branch2(input_img)
        x = torch.cat([o1, o2], dim=1)
        return x


class Inceptionx3(torch.nn.Module):
    def __init__(self, in_fts, out_fts):
        super(Inceptionx3, self).__init__()
        self.branch1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_fts, out_channels=out_fts[0], kernel_size=(
                1, 1), stride=(1, 1)),
            torch.nn.Conv2d(in_channels=out_fts[0], out_channels=out_fts[0], kernel_size=(
                3, 3), stride=(1, 1), padding=1),
            torch.nn.Conv2d(in_channels=out_fts[0], out_channels=out_fts[0], kernel_size=(
                3, 3), stride=(1, 1), padding=1)
        )
        self.branch2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_fts, out_channels=out_fts[1], kernel_size=(
                1, 1), stride=(1, 1)),
            torch.nn.Conv2d(in_channels=out_fts[1], out_channels=out_fts[1], kernel_size=(
                3, 3), stride=(1, 1), padding=1),
        )
        self.branch3 = torch.nn.Sequential(
            torch.nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 1), padding=1),
            torch.nn.Conv2d(in_channels=in_fts, out_channels=out_fts[2], kernel_size=(
                1, 1), stride=(1, 1))
        )
        self.branch4 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_fts, out_channels=out_fts[3], kernel_size=(
                1, 1), stride=(1, 1))
        )

    def forward(self, input_img):
        o1 = self.branch1(input_img)
        o2 = self.branch2(input_img)
        o3 = self.branch3(input_img)
        o4 = self.branch4(input_img)
        x = torch.cat([o1, o2, o3, o4], dim=1)
        return x


class Inceptionx5(torch.nn.Module):
    def __init__(self, in_fts, out_fts, n=7):
        super(Inceptionx5, self).__init__()
        self.branch1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_fts, out_channels=out_fts[0], kernel_size=(
                1, 1), stride=(1, 1)),
            torch.nn.Conv2d(in_channels=out_fts[0], out_channels=out_fts[0], kernel_size=(1, n), stride=(1, 1),
                            padding=(0, n // 2)),
            torch.nn.Conv2d(in_channels=out_fts[0], out_channels=out_fts[0], kernel_size=(n, 1), stride=(1, 1),
                            padding=(n // 2, 0)),
            torch.nn.Conv2d(in_channels=out_fts[0], out_channels=out_fts[0], kernel_size=(1, n), stride=(1, 1),
                            padding=(0, n // 2)),
            torch.nn.Conv2d(in_channels=out_fts[0], out_channels=out_fts[0], kernel_size=(n, 1), stride=(1, 1),
                            padding=(n // 2, 0)),
        )
        self.branch2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_fts, out_channels=out_fts[1], kernel_size=(
                1, 1), stride=(1, 1)),
            torch.nn.Conv2d(in_channels=out_fts[1], out_channels=out_fts[1], kernel_size=(1, n), stride=(1, 1),
                            padding=(0, n // 2)),
            torch.nn.Conv2d(in_channels=out_fts[1], out_channels=out_fts[1], kernel_size=(n, 1), stride=(1, 1),
                            padding=(n // 2, 0)),
        )
        self.branch3 = torch.nn.Sequential(
            torch.nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 1), padding=1),
            torch.nn.Conv2d(in_channels=in_fts, out_channels=out_fts[2], kernel_size=(
                1, 1), stride=(1, 1))
        )
        self.branch4 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_fts, out_channels=out_fts[3], kernel_size=(
                1, 1), stride=(1, 1))
        )

    def forward(self, input_img):
        o1 = self.branch1(input_img)
        o2 = self.branch2(input_img)
        o3 = self.branch3(input_img)
        o4 = self.branch4(input_img)
        x = torch.cat([o1, o2, o3, o4], dim=1)
        return x


class Inceptionx2(torch.nn.Module):
    def __init__(self, in_fts, out_fts):
        super(Inceptionx2, self).__init__()
        self.branch1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_fts,
                            out_channels=out_fts[0] // 4, kernel_size=(1, 1)),
            torch.nn.Conv2d(in_channels=out_fts[0] // 4, out_channels=out_fts[0] // 4, kernel_size=(3, 3), stride=(1, 1),
                            padding=1)
        )
        self.subbranch1_1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=out_fts[0] // 4, out_channels=out_fts[0], kernel_size=(1, 3), stride=(1, 1),
                            padding=(0, 3 // 2))
        )
        self.subbranch1_2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=out_fts[0] // 4, out_channels=out_fts[1], kernel_size=(3, 1), stride=(1, 1),
                            padding=(3 // 2, 0))
        )
        self.branch2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_fts,
                            out_channels=out_fts[2] // 4, kernel_size=(1, 1))
        )
        self.subbranch2_1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=out_fts[2] // 4, out_channels=out_fts[2], kernel_size=(1, 3), stride=(1, 1),
                            padding=(0, 3 // 2))
        )
        self.subbranch2_2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=out_fts[2] // 4, out_channels=out_fts[3], kernel_size=(3, 1), stride=(1, 1),
                            padding=(3 // 2, 0))
        )
        self.branch3 = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=1),
            torch.nn.Conv2d(in_channels=in_fts, out_channels=out_fts[4], kernel_size=(
                1, 1), stride=(1, 1))
        )
        self.branch4 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_fts, out_channels=out_fts[5], kernel_size=(
                1, 1), stride=(1, 1))
        )

    def forward(self, input_img):
        o1 = self.branch1(input_img)
        o11 = self.subbranch1_1(o1)
        o12 = self.subbranch1_2(o1)
        o2 = self.branch2(input_img)
        o21 = self.subbranch2_1(o2)
        o22 = self.subbranch2_2(o2)
        o3 = self.branch3(input_img)
        o4 = self.branch4(input_img)
        x = torch.cat([o11, o12, o21, o22, o3, o4], dim=1)
        return x


class AuxClassifier(torch.nn.Module):
    def __init__(self, in_fts, num_classes):
        super(AuxClassifier, self).__init__()
        self.pool = torch.nn.AdaptiveAvgPool2d(output_size=(5, 5))
        self.conv = torch.nn.Conv2d(in_channels=in_fts,
                                    out_channels=128, kernel_size=(1, 1))
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(5 * 5 * 128, 1024),
            torch.nn.BatchNorm1d(num_features=1024),
            torch.nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        N = x.shape[0]
        x = self.pool(x)
        x = self.conv(x)
        x = x.reshape(N, -1)
        x = self.classifier(x)
        return x


class Inception_v2(torch.nn.Module):
    def __init__(self, in_fts=1, num_classes=10):
        super(Inception_v2, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_fts, out_channels=32,
                            kernel_size=(3, 3), stride=(2, 2)),
            torch.nn.BatchNorm2d(num_features=32)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=32, out_channels=32,
                            kernel_size=(3, 3), stride=(1, 1)),
            torch.nn.BatchNorm2d(num_features=32)
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=32, out_channels=64,
                            kernel_size=(3, 3), stride=(1, 1), padding=1),
            torch.nn.BatchNorm2d(num_features=64)
        )
        self.pool = torch.nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        self.conv4 = torch.nn.Conv2d(
            in_channels=64, out_channels=80, kernel_size=(3, 3), stride=(1, 1))
        self.conv5 = torch.nn.Conv2d(
            in_channels=80, out_channels=192, kernel_size=(3, 3), stride=(2, 2))
        self.conv6 = torch.nn.Conv2d(in_channels=192, out_channels=288, kernel_size=(
            3, 3), stride=(1, 1), padding=1)

        list_incept = [Inceptionx3(in_fts=288, out_fts=[96, 96, 96, 96]),
                       Inceptionx3(in_fts=4 * 96, out_fts=[96, 96, 96, 96]),
                       Inceptionx3(in_fts=4 * 96, out_fts=[96, 96, 96, 96])]

        self.inceptx3 = torch.nn.Sequential(*list_incept)
        self.grid_redn_1 = GridReduction(in_fts=4 * 96, out_fts=384)
        self.aux_classifier = AuxClassifier(768, num_classes)

        list_incept = [Inceptionx5(in_fts=768, out_fts=[160, 160, 160, 160]),
                       Inceptionx5(in_fts=4 * 160,
                                   out_fts=[160, 160, 160, 160]),
                       Inceptionx5(in_fts=4 * 160,
                                   out_fts=[160, 160, 160, 160]),
                       Inceptionx5(in_fts=4 * 160,
                                   out_fts=[160, 160, 160, 160]),
                       Inceptionx5(in_fts=4 * 160, out_fts=[160, 160, 160, 160])]

        self.inceptx5 = torch.nn.Sequential(*list_incept)
        self.grid_redn_2 = GridReduction(in_fts=4 * 160, out_fts=640)

        list_incept = [Inceptionx2(in_fts=1280, out_fts=[256, 256, 192, 192, 64, 64]),
                       Inceptionx2(in_fts=1024, out_fts=[384, 384, 384, 384, 256, 256])]

        self.inceptx2 = torch.nn.Sequential(*list_incept)
        self.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = torch.nn.Linear(2048, num_classes)

    def forward(self, input_img):
        N = input_img.shape[0]
        x = self.conv1(input_img)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.inceptx3(x)
        x = self.grid_redn_1(x)
        aux_out = self.aux_classifier(x)
        x = self.inceptx5(x)
        x = self.grid_redn_2(x)
        x = self.inceptx2(x)
        x = self.avgpool(x)
        x = x.reshape(N, -1)
        x = self.fc(x)
        if self.training:
            return [x, aux_out]
        else:
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
    model = Inception_v2()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
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
            primary_output, secondary_output = model(data)
            print(torch.nn.functional.softmax(primary_output))
            print(torch.nn.functional.softmax(secondary_output))
            input()
            loss = criterion(primary_output, target) + 0.5 * \
                criterion(secondary_output, target)
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
    model = Inception_v2()
    model.load_state_dict(torch.load('models/' + current_file+'_model.pth'))


if __name__ == '__main__':
    train_dataset = MnistDataset(root='./data', train=True)
    test_dataset = MnistDataset(root='./data', train=False)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=False)

    train(train_loader, test_loader)
