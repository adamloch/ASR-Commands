import torch.nn as nn
import torch.optim as optim
import torch
import torchvision
import torchvision.transforms as transforms
from model_gen import Net
from dataset import ASR_Dataset


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
net = Net()
net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)

dataset = ASR_Dataset('validation.csv', '/home/adam/data-asr/png/')
dataset_test = ASR_Dataset('testing.csv', '/home/adam/data-asr/png/')

trainloader = torch.utils.data.DataLoader(
    dataset, batch_size=4, shuffle=True, num_workers=4)
testloader = torch.utils.data.DataLoader(
    dataset_test, batch_size=4, shuffle=True, num_workers=4)

classes = dataset.labels
best = 0
prevacc = -1

for epoch in range(1000):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):

        
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 100 == 99:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f,' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))
    acc = 100 * correct / total

