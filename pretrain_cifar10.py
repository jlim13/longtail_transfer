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
from sklearn.manifold import TSNE
from datasets import cifar10
import matplotlib.pyplot as plt
import matplotlib.cm as cm

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

def view_centers(C, model, dataloader, num_classes = 10, fname = 'samples.png'):

    """
    C is a dict where the key is the class id and the value is the class mean
    """

    color_dict = {}
    color_list = cm.rainbow(np.linspace(0, 1, num_classes))
    for class_id in range(num_classes):
        color_dict[class_id] = color_list[class_id]


    encoded_samples = []
    targets = []
    with torch.no_grad():

        for idx, (img, labels) in enumerate(dataloader):
            img = img.to(device)
            feats = model(img)
            encoded_samples.append(feats.detach().cpu().numpy())
            targets.extend(labels.detach().numpy())
            if idx == 20:
                break

    num_embedded_samples = len(np.vstack(encoded_samples))
    centers = []
    class_center_ids = []
    for idx, (class_id, center) in enumerate(C.items()):
        center = center.detach().cpu().numpy()
        centers.append(center)
        class_center_ids.append(num_embedded_samples + idx )
        targets.append(num_embedded_samples + idx )

    encoded_samples = np.vstack((np.vstack(encoded_samples), np.vstack(centers)))
    targets = np.asarray(targets)

    assert (class_center_ids[-1] == len(encoded_samples)-1)

    samples_embedded = TSNE(n_components=2).fit_transform(encoded_samples)

    for target in range(num_classes):
        label_idxs = (target == targets)
        these_pts = samples_embedded[label_idxs]
        xs = these_pts[:,0]
        ys = these_pts[:,1]

        color = color_dict[target]
        colors = [color] * len(ys)

        plt.scatter(xs, ys, c = colors, alpha = 0.04)

    for idx, class_center_id in enumerate(class_center_ids):
        class_center_embedding = samples_embedded[class_center_id]
        x, y = class_center_embedding

        color = [color_dict[idx]]
        plt.scatter(x, y, c = color, alpha = 1.0)
        plt.annotate('Center_{}'.format(idx), (x,y) )

    plt.savefig(fname)
    plt.clf()




def update_Stats(model, dataloader, minority_class_labels, num_classes = 10):

    #C, Q
    model.eval()

    C = {}
    V = 0

    with torch.no_grad():
        for class_id in range(num_classes):

            #calculate the class centers
            class_running_sum = 0
            class_ct = 0
            for i, (imgs, labels) in enumerate(dataloader):

                relevant_idx = (labels == class_id)
                relevant_images = imgs[relevant_idx]
                relevant_images = relevant_images.to(device)
                batch_size = relevant_images.shape[0]
                if batch_size == 0:
                    continue
                class_ct += batch_size
                relevant_feats = model(relevant_images)

                relevant_feats = relevant_feats.view(batch_size, -1)
                class_running_sum += torch.sum(relevant_feats, axis = 0)

            C[class_id] = class_running_sum / class_ct

        for class_id in range(num_classes):
            if not class_id in minority_class_labels:
                for i, (imgs, labels) in enumerate(dataloader):
                    relevant_idx = (labels == class_id)
                    relevant_images = imgs[relevant_idx]
                    relevant_images = relevant_images.to(device)
                    batch_size = relevant_images.shape[0]
                    if batch_size == 0:
                        continue

                    relevant_feats = model(relevant_images)

                    feat_sub_classCenter = torch.sub(relevant_feats, C[class_id]).unsqueeze(2)
                    feat_sub_classCenter_T = feat_sub_classCenter.permute(0,2,1)
                    v = torch.matmul(feat_sub_classCenter, feat_sub_classCenter_T)

                    v_sum = torch.sum(v, dim=0)
                    V += v_sum

    e, v = torch.eig(V, eigenvectors=True)
    sorted_idx = torch.argsort(e, dim=0, descending=True)[:,0]
    e, v = e[sorted_idx], v[sorted_idx]

    Q = v[:150]
    # torch.save(Q, 'Q.pt')
    # torch.save(C, 'C.pt')
    model.train()

    return Q, C

def transfer_feats(feats, labels, Q, C, minority_labels, device):

    #
    feature_shape = feats.shape
    batch_size = labels.shape[0]

    minority_class_idx = [i for i, elm in enumerate(labels) if elm in minority_labels]
    majority_class_idx = [i for i, elm in enumerate(labels) if not elm in minority_labels]

    # minority_feats = feats[minority_class_idx]
    majority_feats = feats[majority_class_idx]
    majority_feats = majority_feats.view(len(majority_feats), -1)

    majority_labels = labels[majority_class_idx]
    # minority_labels = labels[minority_class_idx].detach().cpu().numpy()

    #get all class centers for majority labels

    transfered_feats = []
    transfered_labels = []

    for maj_label, maj_feat in zip(majority_labels, majority_feats):

        maj_class_center = C[maj_label.item()]
        #sample from minority labels
        minority_label = random.choice(minority_labels)
        min_class_center = C[minority_label]

        QQT = torch.matmul(Q.permute(1,0), Q)
        transfered_feat = min_class_center + torch.matmul(QQT, (maj_feat - maj_class_center) ).unsqueeze(0)
        # transfered_feat = (min_class_center + (maj_feat - maj_class_center) ).unsqueeze(0)

        transfered_feats.append(transfered_feat)
        transfered_labels.append(minority_label)

    transfered_feats = torch.cat(transfered_feats, dim=0)
    transfered_labels = torch.tensor(transfered_labels).long().to(device)

    return transfered_feats, transfered_labels

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

    parser.add_argument("--save_dir", type = str, default = 'weights/pretrain_cifar10_imbalanced/' )
    parser.add_argument("--data_root", type = str, default = './data')
    #model hyperparameters
    #optimizer hyperparameters
    parser.add_argument("--lr", type = float, default = 0.001)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument("--num_epochs", type = int, default = 100)
    parser.add_argument("--num_workers", type = int, default = 8)
    #training helpers
    args = parser.parse_args()

    minority_class_labels = [0, 1, 2, 3, 4, 5]

    train_transform = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                    ])
    test_transform = transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                    ])

    trainset = cifar10.CIFAR10(root='./data',
                        train=True,
                        download=True,
                        transform=train_transform,
                        minority_classes = minority_class_labels,
                        keep_ratio = 0.05)
    print (len(trainset))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=args.num_workers)

    testset = cifar10.CIFAR10(root=args.data_root,
                                        train=False,
                                        download=True,
                                        transform=test_transform,
                                        minority_classes = None,
                                        keep_ratio = None)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=args.num_workers)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)


    backbone = torchvision.models.resnet18(pretrained=False)
    backbone.fc = Identity()
    head = torch.nn.Linear(512, 10)
    backbone.to(device)
    head.to(device)

    cls_criterion = torch.nn.CrossEntropyLoss()
    optimizer_cls = torch.optim.Adam(itertools.chain(backbone.parameters(),
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

        Q, C = update_Stats(backbone, trainloader, minority_class_labels)
        view_centers(C, backbone, testloader, fname = '{}_centers.png'.format(epoch_iter))

        for iter, (data, labels) in enumerate(trainloader):

            optimizer_cls.zero_grad()

            data = data.to(device)
            labels = labels.to(device)

            feats = backbone(data)
            preds = head(feats)

            transfered_feats, transfered_labels = transfer_feats(feats, labels, Q, C, minority_class_labels, device)

            cls_loss = cls_criterion(preds, labels)
            transfer_cls_loss = cls_criterion(head(transfered_feats), transfered_labels)

            loss = cls_loss + transfer_cls_loss

            loss.backward()
            optimizer_cls.step()

            running_epoch_loss += cls_loss.item()

        epoch_loss = running_epoch_loss / len(trainloader)

        accuracy = validate(testloader, backbone, head)

        print ("|Epoch: {} | Epoch Loss: {} | Val Accuracy: {}| Best Accuracy: {}|".format(epoch_iter+1, epoch_loss, accuracy, best_accuracy))

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            encoder_f = 'best_encoder_pretrain.pth'
            cls_f = 'best_cls_pretrain.pth'

            encoder_path = os.path.join(args.save_dir, encoder_f)
            cls_path = os.path.join(args.save_dir, cls_f)

            torch.save(backbone.state_dict(), encoder_path)
            torch.save(head.state_dict(), cls_path)
