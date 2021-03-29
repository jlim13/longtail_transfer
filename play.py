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
from torchvision import datasets
import matplotlib.pyplot as plt

from datasets import cifar10_mixup
import networks
import training_utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RAND_SEED = 17
torch.manual_seed(RAND_SEED)
torch.cuda.manual_seed(RAND_SEED)
np.random.seed(RAND_SEED)
random.seed(RAND_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def validate(val_dataloader, encoder):

    correct = 0
    total = 0

    encoder.eval()

    with torch.no_grad():

        for iter, (data, labels) in enumerate(val_dataloader):
            data = data.to(device)
            labels = labels.to(device)

            preds = encoder(data)

            _, predicted = torch.max(preds.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total

    encoder.train()

    return accuracy

def view_norm(model, epoch_iter):

    class2idx = {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3,
                    'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7,
                    'ship': 8, 'truck': 9}
    idx2class = { v:k for k,v in class2idx.items()}
    norm_dict = {}

    plt.clf()

    fc_weights = model.fc.weight
    with torch.no_grad():

        for idx, matrix in enumerate(fc_weights):
            norm = torch.norm(matrix).item()
            class_name = idx2class[idx]
            norm_dict[class_name] = norm

            plt.scatter(class_name, norm, c = 'b')
    plt.savefig('{}_norm.png'.format(epoch_iter))

def plot_tsne(real_data, synthetic_data, encoder, filename):

    with torch.no_grad():

        encoder.eval()

        encoded_tensors = []
        labels = [] #0 is real, 1 is synthetic
        color_dict = {0: 'r', 1: 'b'}

        for x in real_data:

            real_vid, _, _ = x
            batch_size = real_vid.shape[0]

            real_encoded, _= encoder(real_vid) #, raw_vids = True)
            # real_encoded = real_encoded.mean(dim=1).detach().cpu().numpy()
            # real_encoded = real_encoded.max(dim=1)[0].detach().cpu().numpy()
            real_encoded = real_encoded[:,0].detach().cpu().numpy()
            encoded_tensors.append(real_encoded)
            labels.append([0] * batch_size)

        syn_ct = 0
        for idx, x in enumerate(synthetic_data):

            if idx < 5: continue

            syn_vid, _, _ = x
            batch_size = syn_vid.shape[0]
            syn_encoded, _ = encoder(syn_vid) #, raw_vids = False)
            # syn_encoded = syn_encoded.mean(dim=1).detach().cpu().numpy()
            # syn_encoded = syn_encoded.max(dim=1)[0].detach().cpu().numpy()
            syn_encoded = syn_encoded[:,0].detach().cpu().numpy()
            encoded_tensors.append(syn_encoded)
            labels.append([1] * batch_size)

            syn_ct += batch_size

            if syn_ct > 100:
                break

        encoded_tensors = np.vstack(encoded_tensors)
        labels = np.asarray([item for sublist in labels for item in sublist])

        X_embedded = TSNE().fit_transform(encoded_tensors)
        # encoded_tensors = scaler.fit_transform(encoded_tensors)
        # X_embedded = PCA().fit_transform(encoded_tensors)

        for label in np.unique(labels):

            label_idxs = (label == labels)
            color = color_dict[label]

            these_pts = X_embedded[label_idxs]

            xs = these_pts[:,0]
            ys = these_pts[:,1]

            colors = [color] * len(ys)
            if label == 0:
                label_plt = 'Real'
            else:
                label_plt = 'Synthetic'
            plt.scatter(xs, ys, c = colors, label=label_plt)
            plt.legend(loc = 'best')
            plt.xticks([]),plt.yticks([])



            # if label == 0:
            #     fname = '{}_real_pts_latex.txt'.format(filename)
            # else:
            #     fname = '{}_syn_pts_latex.txt'.format(filename)
            #
            # with open(fname, 'w+') as f:
            #
            #     f.write('x y\n')
            #     for x, y in zip(xs, ys):
            #         f.write(str(x) + ' ' +str(y) + '\n')


        plt.savefig(filename)
        plt.clf()



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--save_dir", type = str, default = 'images_vanilla_gan/' )
    parser.add_argument("--data_root", type = str, default = 'data/')

    #model hyperparameters
    #optimizer hyperparameters
    parser.add_argument("--lr", type = float, default = 0.005)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument("--num_epochs", type = int, default = 100)
    parser.add_argument("--num_workers", type = int, default = 8)
    parser.add_argument("--num_classes", type = int, default = 10)
    #training helpers
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    train_transform = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ])

    trainset = cifar10_mixup.Pure_CIFAR10(root=args.data_root,
                        train=True,
                        transform=train_transform,
                        target_transform = None,
                        minority_class = 'automobile',
                        keep_ratio = 1.0)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=args.num_workers)

    testset = cifar10_mixup.Pure_CIFAR10(root=args.data_root,
                            train=False,
                            transform=train_transform,
                            minority_class = None,
                            keep_ratio = None)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=args.num_workers)

    cls = torchvision.models.resnet18(pretrained=False)
    cls.fc = torch.nn.Linear(512, 10)
    cls.to(device)

    cls_criterion = torch.nn.CrossEntropyLoss()
    optimizer_cls = torch.optim.Adam(cls.parameters(), lr = args.lr)

    best_accuracy = -1

    for epoch_iter in range(args.num_epochs):

        running_epoch_loss = 0

        for iter, (data, labels) in enumerate(trainloader):

            optimizer_cls.zero_grad()

            data = data.to(device)
            labels = labels.to(device)

            preds = cls(data)
            cls_loss = cls_criterion(preds, labels)

            cls_loss.backward()
            optimizer_cls.step()

            running_epoch_loss += cls_loss.item()

        epoch_loss = running_epoch_loss / len(trainloader)

        accuracy = validate(testloader, cls)
        view_norm(cls, epoch_iter)
        print ("|Epoch: {} | Epoch Loss: {} | Val Accuracy: {}|".format(epoch_iter+1, epoch_loss, accuracy))

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            # encoder_f = 'best_encoder_pretrain.pth'
            # cls_f = 'best_cls_pretrain.pth'
            #
            # encoder_path = os.path.join(args.save_dir, encoder_f)
            #
            # torch.save(encoder.state_dict(), encoder_path)
