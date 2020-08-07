import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions.binomial as binomial


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(5, 4)
        self.fc2 = nn.Linear(4, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


true_proba = 0.8
base_input = torch.tensor([0,0,0,0,1]).unsqueeze(dim=0).repeat(80, 1).float()
print("base_input", base_input)
rd_input = (lambda _:base_input + torch.randn(size=(80,1))/10)

for batch in range(5000):  # loop over the dataset multiple times

    current_true_proba = true_proba - true_proba * batch / 5000
    binamoal_sampler = binomial.Binomial(1, torch.tensor([current_true_proba]))
    inputs, labels = rd_input(0), binamoal_sampler.sample(sample_shape=(80,))

    optimizer.zero_grad()

    outputs = net(inputs)

    labels= labels.long().squeeze(dim=-1)
    # print("outputs, labels.argmax(dim=1)", outputs, labels)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    outputs = torch.nn.functional.softmax(outputs)
    print('outputs', outputs, "current_true_proba", current_true_proba)

print('Finished Training')