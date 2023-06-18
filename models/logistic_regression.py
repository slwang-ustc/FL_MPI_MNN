from __future__ import print_function

import MNN
nn = MNN.nn
F = MNN.expr


class LR(nn.Module):
    def __init__(self, input_size):
        super(LR, self).__init__()
        self.fc1 = nn.linear(input_size, 1024)
        self.fc2 = nn.linear(1024, 128)
        self.fc3 = nn.linear(128, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x, 1)
        return x
