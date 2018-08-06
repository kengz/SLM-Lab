import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Part 1 intro https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html#sphx-glr-beginner-blitz-tensor-tutorial-py
torch.empty(5, 3)
torch.empty((5, 3))
torch.rand(5, 3)
torch.zeros(5, 3, dtype=torch.long)
x = torch.tensor([5, 5, 3])
x

x = x.new_ones(5, 3, dtype=torch.double)
x = torch.rand_like(x, dtype=torch.float)
x

x.size()

y = torch.rand(5, 3)
x + y
torch.add(x, y)

result = torch.empty(5, 3)
torch.add(x, y, out=result)
result

y.add_(x)

x[:, 1]

x = torch.rand(4, 4)
x.size()
y = x.view(16)
y.size()
z = x.view(-1, 8)
z.size()

x = torch.rand(1)
x.item()


# share memory with numpy
a = torch.ones(5)
a

b = a.numpy()
b
a.add_(1)
a
b

a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
a
b

if torch.cuda.is_available():
    device = torch.device('cuda')
    y = torch.ones_like(x, device=device)
    x = x.to(device)
    z = x + y
    print(z)
    print(z.to('cpu', torch.double))


# Part 2 autograd https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html

# Tensors auto-track compute history with `.requires_grad=True`, then call `.backward()` to do autograd
# to stop tracking history, call `.clone()`.
# to prevent tracking history (and using memory) altogether, use in `with torch.no_grad():` super useful for eval model
# If tensor is scalar, no arg for `backward()`, else supply argument that is a tensor of matching shape

x = torch.ones(2, 2, requires_grad=True)
x

y = x + 2
y  # result of an operation, so has a `grad_fn`
y.grad_fn

z = y * y * 3
out = z.mean()
z.grad_fn
out.grad_fn


a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
a.requires_grad
a.requires_grad_(True)
a.requires_grad
b = (a * a).sum()
b.grad_fn


# gradients
out
out.backward()
x.grad  # d(out)/dx

x = torch.randn(3, requires_grad=True)
y = x * 2
while y.data.norm() < 1000:
    y = y * 2
y

# gradients = x values to apply after dy/dx
gradients = torch.tensor([0.1, 1.0, 0.001], dtype=torch.float)
y.backward(gradients)
x.grad


x.requires_grad
(x ** 2).requires_grad

with torch.no_grad():
    print((x ** 2).requires_grad)


# Part 3 NN https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        # 1 input channel, 6 output channels, 5x5 square conv kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # affine op y = Wx+b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # max pooling over (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # use 1 number for square
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dims except batch dim
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)
# `backward()` is automatically defined with autograd
params = list(net.parameters())
print(len(params))
print(params[0].size())
print(params[1].size())
print(params[2].size())
print(params[3].size())

input = torch.randn(1, 1, 32, 32)
out = net(input)
out
# zero gradient buffer of all params and backprop with random gradients
net.zero_grad()
out.backward(torch.rand(1, 10))


# Loss
output = net(input)
output
target = torch.arange(1, 11)
target
target = target.unsqueeze(0)  # make same shape as input
target
criterion = nn.MSELoss()

loss = criterion(output, target)
loss
loss.grad_fn
loss.grad_fn.next_functions[0][0]
loss.grad_fn.next_functions[0][0].next_functions[0][0]

# need to clear or else gradients will be accumulated
net.zero_grad()
net.conv1.bias.grad  # shd be done after clearing, ok

loss.backward()

net.conv1.bias.grad

# update weights, use weight = weight - learning_rate * gradient
learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)


# or use other update rules like SGD, Adam etc optim:
import torch.optim as optim
optimizer = optim.SGD(net.parameters(), lr=0.01)

# in training loop
optimizer.zero_grad()
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()  # do the update using loss-computed grad


# Part 4 data https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

import torchvision
import torchvision.transforms as transforms

# torchvision data output range is [0, 1]. transform into range [-1, 1]
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')


# eval with `no_grad()`
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))


# Train on GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device

# then just call this to auto put all params and buffers over
net.to(device)

# and remember to send input and output too in training loop
# inputs, labels = inputs.to(device), labels.to(device)
