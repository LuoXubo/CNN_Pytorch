import torch
import torchvision
from torch import nn

test_data = torchvision.datasets.MNIST(
    root='MINIST',
    train=False
)

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


cnn = torch.load('cnn.minist.pkl')
test_x = torch.unsqueeze(test_data.data, dim=1).type(torch.FloatTensor) / 255
test_y = test_data.targets

test_output = cnn(test_x[:20])
pred_y = torch.max(test_output, 1)[1].data.numpy()

print(pred_y, 'prediction number')
print(test_y[:20].numpy(), 'real number')

test_output1 = cnn(test_x)
pred_y1= torch.max(test_output1, 1)[1].data.numpy()
accuracy = float((pred_y1 == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
print('accuracy ', accuracy)
