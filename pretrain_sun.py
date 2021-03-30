import os
import itertools
import random
import argparse
import torch
import numpy as np
import torchvision
import sys
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import sun
import warnings
import training_utils

warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RAND_SEED = 1
torch.manual_seed(RAND_SEED)
torch.cuda.manual_seed(RAND_SEED)
np.random.seed(RAND_SEED)
random.seed(RAND_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def validate(val_dataloader, backbone, head):

    correct = 0
    total = 0

    backbone.eval()

    with torch.no_grad():

        for iter, (data, labels) in enumerate(val_dataloader):
            data = data.to(device)
            labels = labels.to(device)

            feats = backbone(data)
            preds = head(feats)

            _, predicted = torch.max(preds.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total

    backbone.train()

    return accuracy

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--save_dir", type = str, default = 'weights/sun_imbalanced/' )
    parser.add_argument("--data_root", type = str, default = './data')
    #model hyperparameters
    #optimizer hyperparameters
    parser.add_argument("--lr", type = float, default = 0.01)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument("--num_epochs", type = int, default = 100)
    parser.add_argument("--num_workers", type = int, default = 8)
    #training helpers
    args = parser.parse_args()

    train_transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(224),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    test_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    trainset = sun.SunDataset(data_root = '/data/', data_txt_file = '/data/sun397_train_lt.txt', transform = train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=args.num_workers)
    testset = sun.SunDataset(data_root = '/data/', data_txt_file = '/data/sun397_test_lt.txt', transform = test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=args.num_workers)

    minority_class_labels = trainset.tail_classes

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)


    backbone = torchvision.models.resnet152(pretrained=True)
    backbone.fc = Identity()
    head = torch.nn.Linear(2048, 397)
    backbone.to(device)
    head.to(device)

    backbone = torch.nn.DataParallel(backbone)
    head = torch.nn.DataParallel(head)

    cls_criterion = torch.nn.CrossEntropyLoss()
    optimizer_cls = torch.optim.SGD(itertools.chain(backbone.parameters(),
                                            head.parameters()), lr = args.lr)

    best_accuracy = -1

    encoder_f = 'best_encoder_pretrain.pth'
    cls_f = 'best_cls_pretrain.pth'
    encoder_path = os.path.join(args.save_dir, encoder_f)
    cls_path = os.path.join(args.save_dir, cls_f)
    backbone.load_state_dict(torch.load(encoder_path))
    head.load_state_dict(torch.load(cls_path))

    for epoch_iter in range(args.num_epochs):

        running_epoch_loss = 0

        Q, C = training_utils.update_Stats(backbone, trainloader, minority_class_labels, device)
        # view_centers(C, backbone, testloader, fname = '{}_centers.png'.format(epoch_iter))

        for iter, (data, labels) in enumerate(trainloader):

            optimizer_cls.zero_grad()

            data = data.to(device)
            labels = labels.to(device)

            feats = backbone(data)
            preds = head(feats)

            transfered_feats, transfered_labels = training_utils.transfer_feats(feats, labels, Q, C, minority_class_labels, device)

            transfer_cls_loss = cls_criterion(head(transfered_feats), transfered_labels)
            cls_loss = cls_criterion(preds, labels)

            loss = cls_loss + transfer_cls_loss

            loss.backward()
            optimizer_cls.step()

            running_epoch_loss += cls_loss.item()

        epoch_loss = running_epoch_loss / len(trainloader)

        accuracy = validate(testloader, backbone, head)


        if accuracy > best_accuracy:
            best_accuracy = accuracy
            encoder_f = 'best_encoder_pretrain.pth'
            cls_f = 'best_cls_pretrain.pth'

            encoder_path = os.path.join(args.save_dir, encoder_f)
            cls_path = os.path.join(args.save_dir, cls_f)

            torch.save(backbone.state_dict(), encoder_path)
            torch.save(head.state_dict(), cls_path)

        print ("|Epoch: {} | Epoch Loss: {} | Val Accuracy: {}| Best Accuracy: {}|".format(epoch_iter+1, epoch_loss, accuracy, best_accuracy))
