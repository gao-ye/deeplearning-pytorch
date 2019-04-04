import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from utils import progress_bar
from tensorboardX import SummaryWriter


os.environ["CUDA_VISIBLE_DEVICES"] = "6"

def argsparse():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--epoch', default=200, type=int, help='epoch size')

    parser.add_argument('--resume', '-r', action='store_true',default=False,help='resume from checkpoint')

    args = parser.parse_args()
    return args

def load_train_data():

    transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    ## if you cannot run the program because of "OUT OF MEMORY", you can decrease the batch_size properly.
    trainset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, 
        batch_size=200, 
        shuffle=True, 
        num_workers=2)

    return trainloader

def load_test_data():

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    testset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform_test)

    testloader = torch.utils.data.DataLoader(
        testset, 
        batch_size=200, 
        shuffle=False, 
        num_workers=2)

    return testloader

class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)

        self.bn1 = nn.BatchNorm2d(6)
        self.dp1 = nn.Dropout(0.5)


        self.conv2 = nn.Conv2d(6, 16, 5)

        self.bn2 = nn.BatchNorm2d(16)
        self.dp2 = nn.Dropout(0.5)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))

        x = self.bn1(x)
        x = self.dp1(x)

        x = self.pool(F.relu(self.conv2(x)))

        x = self.bn2(x)
        x = self.dp2(x)
        # fully connect
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size = 3, padding=1),
            # nn.Dropout(0.5),
            nn.BatchNorm2d(16),
            nn.ReLU()
            )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            # nn.Dropout(0.5),
            nn.BatchNorm2d(32),
            nn.ReLU()
            )

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            # nn.Dropout(0.5),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
            )

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            # nn.Dropout(0.5),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
            )

        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            # nn.Dropout(0.5),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # nn.MaxPool2d(2)
            )
        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            # nn.Dropout(0.5),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2)
            )

        self.fc1 = nn.Linear(4*4*512, 120)
        # self.dp1 = nn.Dropout(0.5)
        # self.fc2 = nn.Linear(120, 84)
        # self.dp2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(120, 10)

        self.softmax = nn.Softmax(dim =1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        # print("after layer 2 {0}".format(out.shape))
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)

        # print("after layer 3 {0}".format(out.shape))
        out = out.view(out.size(0), -1)
        out =  F.relu(self.fc1(out))
        # out = self.dp1(out)
        out =  F.relu(self.fc2(out))
        # out = self.dp2(out)
        # out = F.relu(self.fc3(out))
        out = self.softmax(out)

        return out


def train(net, criterion, optimizer, trainloader, n):
    device = 'cuda'
    print('Epoch: {0}'.format(n))

    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return 100.0*correct/total

def test(net, criterion, optimizer, testloader, n):
    device = 'cuda'

    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    print('Epoch: {0} test acc {1}'.format(n, acc))
    return acc

    # if acc > best_acc:
    #     print('Saving..')
    #     state = {
    #         'net': net.state_dict(),
    #         'acc': acc,
    #         'epoch': epoch,
    #     }
    #     if not os.path.isdir('checkpoint'):
    #         os.mkdir('checkpoint')
    #     if not os.path.isdir('checkpoint/'+netname):
    #         os.mkdir('checkpoint/'+netname)
    #     torch.save(state, './checkpoint/'+netname + '/ckpt.t7')
    #     best_acc = acc

def adjust_lr_exp(optimizer, new_lr):
    for g in optimizer.param_groups:
        g['lr'] = new_lr
    print("learning rate now is {}".format(new_lr))

if __name__ == "__main__":

    args = argsparse()
    print(args)

    print("Preparing data ===>")
    trainloader = load_train_data()
    testloader  = load_test_data()
    print('EPOCH:', len(trainloader), len(testloader))

    print("building the model ===>")
    net = Net()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    best_acc = 0
    writer = SummaryWriter('runs/cifar-10')
    import time
    start = time.clock()

    LR = []
    for i in range(20):
        LR.append(0.1)   
    for i in range(15):
        LR.append(0.05)
    for i in range(5):
        LR.append(0.01)
    for i in range(40):
        LR.append(0.005)
    for i in range(50):
        LR.append(0.001)
    # print(LR)
        
    for n in  range(args.epoch):
        if n < len(LR):
            new_lr = LR[n]
            # print(" <----new_lr is {}".format(new_lr))
        else:
            new_lr = 1e-4
            # print(" >----new_lr is {}".format(new_lr))

        adjust_lr_exp(optimizer, new_lr)
        # for g in optimizer.param_groups:
        #     g['lr'] = new_lr
        for g in optimizer.param_groups:
            print("g['lr'] is {}".format(g['lr']))
        print("\n")

        tra_acc = train(net, criterion, optimizer, trainloader, n)
        writer.add_scalars("acc", { 'train': tra_acc},  n)

        if n% 1 ==0:
            test_acc= test(net, criterion, optimizer, testloader, n)
            writer.add_scalars("acc", { 'test': test_acc},  n)

    print("train and test over")
    #end time
    end = time.clock()
    second = end-start
    minute = int(second /60)
    second = int(second - minute*60)
    print ("time is  {0} minute {1} second ".format(minute, second))




