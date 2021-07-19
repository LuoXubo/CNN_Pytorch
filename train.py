import torch
import torch.nn as nn
import torchvision
import torch.utils.data as Data

EPOCH = 3
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD_MINIST = False

train_data = torchvision.datasets.MNIST(
    root = './MINIST',
    train = True,
    transform = torchvision.transforms.ToTensor(),
    download = DOWNLOAD_MINIST
)

train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_data = torchvision.datasets.MNIST(root='./MINIST', train=False)

test_x = torch.unsqueeze(test_data.data, dim=1).type(torch.FloatTensor)[:2000] / 255
test_y = test_data.targets[:2000]

# torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:2000] / 255

# 搭建CNN
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1
            ),                              # (1,28,28) -> (16,28,28)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)     # (16,28,28) -> (16,14,14)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1
            ),                              # (16,14,14) -> (32,14,14)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)     # (32,14,14) -> (32,7,7)
        )
        self.output = nn.Linear(32*7*7, 10)

    def forward(self, x):
        out = self.conv1(x)                 # (Batch,1,28,28) -> (Batch,16,14,14)
        out = self.conv2(out)               # (Batch,16,14,14) -> (Batch,32,7,7)
        out = out.view(out.size(0), -1)     # (Batch,32,14,14) -> (Batch,32*14*14)
        out = self.output(out)
        return out

cnn = CNN()
# print(cnn)

# Adam优化
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step , (b_x, b_y) in enumerate(train_loader):
        output = cnn(b_x)
        loss = loss_func(output, b_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step%50==0:
            test_output = cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.numpy()
            accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)

torch.save(cnn, 'cnn.minist.pkl')
print('finish training')

