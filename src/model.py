import torch
from torch import nn

class Vege15(nn.Module):
    def __init__(self):
        super(Vege15, self).__init__()
        self.model1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=2),
            nn.MaxPool2d(3),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),  
            nn.Linear(6272, 128),
#             nn.Dropout(0.5),
            nn.Linear(128, 15)
        )

    def forward(self, x):
        x = self.model1(x)
        return x

if __name__ == '__main__':
    vege15 = Vege15()
    input = torch.ones((32, 3, 64, 64))
    output = vege15(input)
    print(output.shape)

