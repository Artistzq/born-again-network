'''Train CIFAR10 with PyTorch.'''
import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar
from utils import get_model as get_model_from_path
from ban_loss import BANLoss

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--teacher', default=None, type=str, help='ban_teacher_path')
parser.add_argument('--save_path', type=str, help='save_path')
parser.add_argument('--ban_start_num', default=0, type=int)
parser.add_argument('--ban_num', default=5, type=int)
parser.add_argument('--epoch_num', default=200, type=int)
parser.add_argument('--save_interval', default=50, type=int)
parser.add_argument('--seed', default=42, type=int)
args = parser.parse_args()

def get_model():
    return get_model_from_path(args.save_path)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if device == 'cuda':
    torch.backends.cudnn.benchmark = True
    
if args.seed is not None:
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    
if not os.path.exists("results"):
    os.mkdir("results")

class Trainer:
    def __init__(self, net, trainloader, testloader, criterion, optimizer, 
                 scheduler, epoch_num, save_interval, save_path, device, 
                 save_best, save_name=None, save_acc=True) -> None:
        self.net = net
        self.trainloader = trainloader
        self.testloader = testloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_path = save_path
        self.device = device
        self.name = "model" if save_name is None else save_name
        self.best_acc = -1
        self.epoch_num = epoch_num
        self.save_interval = save_interval if save_interval <= epoch_num else 1
        self.save_best = save_best
        self.save_acc = save_acc
        
        if not os.path.exists("./results/{}".format(save_path)):
            os.mkdir("./results/{}".format(save_path))
        if self.save_acc:
            f = open("results/{}/{}_acc_loss.log".format(self.save_path, self.name), "w")
            f.close()

    def train(self, epoch):
        print('\nEpoch: %d' % epoch)
        self.net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.net(inputs)
            # print(type(inputs), type(outputs), type(targets))
            loss = self.criterion(outputs, targets, inputs)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(self.trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    def test(self, epoch):
        self.net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.testloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.net(inputs)
                loss = self.criterion(outputs, targets, inputs)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(batch_idx, len(self.testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

        # Save checkpoint.
        acc = 100.*correct/total
        
        if self.save_acc:
            with open("results/{}/{}_acc_loss.log".format(self.save_path, self.name), "a+") as f:
                f.write("epoch:{} test_acc:{:4f} test_loss:{:.6f}\n".format(epoch+1, acc, test_loss))
        
        if self.save_path is not None:
            if not os.path.exists("results/{}".format(self.save_path)):
                os.mkdir("results/{}".format(self.save_path))
            
            if acc > self.best_acc and self.save_best:
                # 保存最好的
                print("Better Acc {}, saving ...".format(acc))
                state = {
                    'net': self.net.state_dict(),
                    'acc': acc,
                    'epoch': epoch,
                }
                torch.save(state, 'results/{}/{}_best.pth'.format(self.save_path, self.name))
                self.best_acc = acc
    
            if (epoch+1) % self.save_interval == 0:
                # 按间隔保存
                print('Saving..')
                state = {
                    'net': self.net.state_dict(),
                    'acc': acc,
                    'epoch': epoch,
                }
                torch.save(state, 'results/{}/{}_ep{}.pth'.format(self.save_path, self.name, epoch+1))


# Data
print('==> Preparing data..')
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

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


def single_train():
    # Model
    print('==> Building model..')

    net = get_model()
    net = net.to(device)

    # Trainer
    criterion = nn.CrossEntropyLoss()

    if args.teacher is not None:
        print("==> teacher: {}, start ban...".format(args.teacher))
        teacher_net = get_model()
        teacher_net.load_state_dict(torch.load(args.teacher)["net"], False)
        teacher_net.to(device)
        criterion = BANLoss(criterion, teacher_net)
    else:
        criterion = BANLoss(criterion, None)

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch_num)

    trainer = Trainer(
        net=net,
        trainloader=trainloader,
        testloader=testloader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch_num=args.epoch_num, 
        save_interval=args.save_interval,
        save_path=args.save_path,
        device=device,
        save_best=True
    )

    start_epoch = 0
    for epoch in range(start_epoch, start_epoch + args.epoch_num):
        trainer.train(epoch)
        trainer.test(epoch)
        trainer.scheduler.step()
    
    return trainer.net
    
def ban_repeat(start, end, init_net):
    net = init_net
    for ban_idx in range(start, end):
        ban_idx = ban_idx + 1
        
        teacher_net = net
        net = get_model().to(device)

        teacher_net.to(device)
        criterion = BANLoss(nn.CrossEntropyLoss(), teacher_net)
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch_num)

        trainer = Trainer(
            net=net,
            trainloader=trainloader,
            testloader=testloader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch_num=args.epoch_num, 
            save_interval=args.save_interval,
            save_path=args.save_path,
            device=device,
            save_name="ban{}".format(ban_idx),
            save_best=True
        )
        
        start_epoch = 0
        for epoch in range(start_epoch, start_epoch + args.epoch_num):
            trainer.train(epoch)
            trainer.test(epoch)
            trainer.scheduler.step()

if __name__ == "__main__":
    if args.teacher is None:
        single_train()
    else:
        teacher_net = get_model()
        teacher_net.load_state_dict(torch.load(args.teacher)["net"], False)
        teacher_net.to(device)
        ban_repeat(args.ban_start_num, args.ban_num, teacher_net)