import os
import random
import argparse
import time
from datetime import datetime
from pytz import timezone
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from PIL import Image
import cv2

# Hyper Parameters
BATCH_SIZE = 50


class ContrastiveLoss(torch.nn.Module):

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, input1, input2, y):
        diff = input1 - input2
        dist_sq = torch.sum(torch.pow(diff, 2), 1)
        dist = torch.sqrt(dist_sq)
        mdist = self.margin - dist
        dist = torch.clamp(mdist, min=0.0)
        loss = y * dist_sq + (1 - y) * torch.pow(dist, 2)
        loss = torch.sum(loss) / 2.0 / input1.size()[0]
        return loss


class LFWDataset(Dataset):

    def __init__(self, root_dir, path_file_dir, transform=None, random_aug=False):
        self.root_dir = root_dir
        path_file = open(path_file_dir, 'r')
        data = []
        for line in path_file:
            line = line.strip()
            img1, img2, label = line.split(' ')
            label = int(label)
            data.append((img1, img2, label))
        self.data = data
        self.transform = transform
        self.random_aug = random_aug
        self.random_aug_prob = 0.7
        path_file.close()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img1, img2, label = self.data[idx]
        img1_file = Image.open(os.path.join(self.root_dir, img1))
        img2_file = Image.open(os.path.join(self.root_dir, img2))
        if self.random_aug:
            img1_file = self.random_augmentation(img1_file, self.random_aug_prob)
            img2_file = self.random_augmentation(img2_file, self.random_aug_prob)

        if self.transform:
            img1_file = self.transform(img1_file)
            img2_file = self.transform(img2_file)
        return (img1_file, img2_file, label)

    def random_augmentation(self, img, prob):
        def rotate(img):
            degree = random.randrange(-30, 30)
            return img.rotate(degree)
        def flip(img):
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        def translate(img):
            d_x = random.randrange(-10, 10)
            d_y = random.randrange(-10, 10)
            img = np.array(img)
            mat = np.float32([[1, 0, d_x], [0, 1, d_y]])
            num_rows, num_cols = img.shape[:2]
            img = cv2.warpAffine(img, mat, (num_cols, num_rows))
            return Image.fromarray(np.uint8(img))
        def scale(img):
            scale = 0.7 + 0.6 * random.random()
            img = np.array(img)
            mat = np.float32([[scale, 0, 0], [0, scale, 0]])
            num_rows, num_cols = img.shape[:2]
            img = cv2.warpAffine(img, mat, (num_cols, num_rows))
            return Image.fromarray(np.uint8(img))

        if random.random() > prob:
            return img

        transform_ops = [rotate, flip, translate, scale]
        op_len = random.randrange(1, len(transform_ops) + 1)
        ops = random.sample(transform_ops, op_len)
        for op in ops:
            img = op(img)
        return img


class Flatten(nn.Module):

    def forward(self, input):
        return input.view(input.size(0), -1)


class SiameseNetwork(nn.Module):

    def __init__(self, contra_loss=False):
        super(SiameseNetwork, self).__init__()

        self.contra_loss = contra_loss

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, padding=2, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=5, padding=2, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),

            Flatten(),
            nn.Linear(131072, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(1024)
        )

        self.fc = nn.Sequential(
            nn.Linear(2048, 1),
            nn.Sigmoid()
        )

    def forward_once(self, x):
        output = self.cnn(x)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        if self.contra_loss:
            return output1, output2
        else:
            output = torch.cat((output1, output2), 1)
            output = self.fc(output)
            return output


def threashold_sigmoid(t):
    """prob > 0.5 --> 1 else 0"""
    threashold = t.clone()
    threashold.data.fill_(0.5)
    return (t > threashold).float()


def threashold_contrastive_loss(input1, input2, m):
    """dist < m --> 1 else 0"""
    diff = input1 - input2
    dist_sq = torch.sum(torch.pow(diff, 2), 1)
    dist = torch.sqrt(dist_sq)
    threashold = dist.clone()
    threashold.data.fill_(m)
    return (dist < threashold).float().view(-1, 1)


def cur_time():
    fmt = '%Y-%m-%d %H:%M:%S %Z%z'
    eastern = timezone('US/Eastern')
    naive_dt = datetime.now()
    loc_dt = datetime.now(eastern)
    return loc_dt.strftime(fmt).replace(' ', '_')


def train(args):
    default_transform = transforms.Compose([
        transforms.Scale(128),
        transforms.ToTensor(),
    ])
    train_dataset = LFWDataset('./lfw', './train.txt', default_transform, args.randaug)
    print("Loaded {} training data.".format(len(train_dataset)))

    # # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=BATCH_SIZE,
                                               shuffle=True)

    siamese_net = SiameseNetwork(args.contra_loss)
    if args.cuda:
        siamese_net = siamese_net.cuda()

    # # Loss and Optimizer
    if args.contra_loss:
        criterion = ContrastiveLoss(margin=args.margin)
    else:
        criterion = nn.BCELoss()

    optimizer = torch.optim.Adam(siamese_net.parameters())

    # Train the Model
    num_epochs = args.epoch
    for epoch in range(num_epochs):
        for i, (img1_set, img2_set, labels) in enumerate(train_loader):

            if args.cuda:
                img1_set = img1_set.cuda()
                img2_set = img2_set.cuda()
                labels = labels.cuda()

            img1_set = Variable(img1_set)
            img2_set = Variable(img2_set)
            labels = Variable(labels.view(-1, 1).float())

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            if args.contra_loss:
                output1, output2 = siamese_net(img1_set, img2_set)
                loss = criterion(output1, output2, labels)
                loss.backward()
                optimizer.step()
            else:
                output_labels_prob = siamese_net(img1_set, img2_set)
                loss = criterion(output_labels_prob, labels)
                loss.backward()
                optimizer.step()
        print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' % (epoch+1, num_epochs, i+1, len(train_dataset)//BATCH_SIZE, loss.data[0]))

    # Training accuracy
    test_against_data(args, 'training', train_loader, siamese_net)

    # Save the Trained Model
    model_file_name = "{}_{}".format(cur_time(), args.model_file)
    torch.save(siamese_net.state_dict(), model_file_name)
    print("Saved model at {}".format(model_file_name))
    return siamese_net


def test_against_data(args, label, dataset, siamese_net):
    # Training accuracy
    siamese_net.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    correct = 0.0
    total = 0.0
    for img1_set, img2_set, labels in dataset:
        labels = labels.view(-1, 1).float()
        if args.cuda:
            img1_set = img1_set.cuda()
            img2_set = img2_set.cuda()
            labels = labels.cuda()
        img1_set = Variable(img1_set)
        img2_set = Variable(img2_set)
        labels = Variable(labels)

        if args.contra_loss:
            output1, output2 = siamese_net(img1_set, img2_set)
            output_labels = threashold_contrastive_loss(output1, output2, args.margin)
        else:
            output_labels_prob = siamese_net(img1_set, img2_set)
            output_labels = threashold_sigmoid(output_labels_prob)

        if args.cuda:
            output_labels = output_labels.cuda()
        total += labels.size(0)
        correct += (output_labels == labels).sum().data[0]

    print('Accuracy of the model on the {} {} images: {} %%'.format(total, label, (100 * correct / total)))


def test(args, siamese_net=None):
    if not siamese_net:
        saved_model = torch.load(args.model_file)
        siamese_net = SiameseNetwork(args.contra_loss)
        siamese_net.load_state_dict(saved_model)

    if args.cuda:
        siamese_net = siamese_net.cuda()

    default_transform = transforms.Compose([
        transforms.Scale(128),
        transforms.ToTensor(),
    ])
    test_dataset = LFWDataset('./lfw', './test.txt', default_transform)
    print("Loaded {} test data.".format(len(test_dataset)))

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=BATCH_SIZE,
                                              shuffle=False)

    test_against_data(args, "testing", test_loader, siamese_net)


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("action", nargs='?', choices=['train', 'test', 'train_test'], default='train_test')
    arg_parser.add_argument("model_file", nargs='?', help="model file path", default='siamese.pkl')
    arg_parser.add_argument("-e", "--epoch", type=int, help="training epochs", default=1)
    arg_parser.add_argument("-m", "--margin", type=float, help="training epochs", default=1.0)
    arg_parser.add_argument("-c", "--cuda", action='store_true', default=False)
    arg_parser.add_argument("-r", "--randaug", action='store_true', default=False)
    arg_parser.add_argument("-cl", "--contra_loss", action='store_true', default=False)

    args = arg_parser.parse_args()

    print("Invoke {} with args {}".format(args.action, args))
    if args.action == "train":
        train(args)
    elif args.action == "test":
        test(args)
    elif args.action == 'train_test':
        siamese_net = train(args)
        test(args, siamese_net)


if __name__ == '__main__':
    main()
