import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.optim.lr_scheduler import MultiStepLR
import argparse
from tqdm import tqdm
from model.resnet import ResNet18


def train(model, train_loader, optimizer, scheduler, num_epochs=45):
    # train
    print("Start training with optimizer ...")
    model.train()
    for epoch in range(num_epochs):
        loss_total = 0.0
        bar = tqdm(train_loader, total=len(train_loader), ncols=0)
        for i, batch_data in enumerate(bar):
            inputs, labels = batch_data[0].cuda(), batch_data[1].cuda()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # pipeline
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_total += loss.item()
            if i % 20 == 19:
                description = 'epoch: %d, iters: %5d, loss: %.3f' % (epoch + 1, i + 1, loss_total / 20)
                bar.set_description(desc=description)
                loss_total = 0.0
        scheduler.step()
    print("Training Finished!")


def test(model: nn.Module, test_loader):
    # test
    model.eval()
    size = len(test_loader.dataset)
    correct = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.cuda()
            labels = labels.cuda()
            output = model(inputs)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(labels.data.view_as(pred)).sum().item()
    print('\nTest set: Accuracy: {}/{} ({:.2f}%)\n'.format(
        correct, size,
        100 * correct / size))


if __name__ == "__main__":
    # construct the model
    model = ResNet18(10).cuda()
    # construct the dataset
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_set = torchvision.datasets.CIFAR10('./data/', train=True, download=True, transform=transform_train)
    test_set = torchvision.datasets.CIFAR10("./data/", train=False, download=True, transform=transform_test)
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = MultiStepLR(optimizer=optimizer, milestones=[15, 30], gamma=0.1)
    train(model, train_loader, optimizer, scheduler)
    test(model, test_loader)
