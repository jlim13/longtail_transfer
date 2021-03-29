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
from torchvision.utils import save_image
import itertools

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

def save_ims_in_batch(batch_data):

    for j, tensor in enumerate(batch_data):
        save_image(tensor, 'img_{}.png'.format(j), normalize = True)

def sample_image(generator, n_row, batches_done, outdir, device, args):
    """Saves a grid of generated digits ranging from 0 to num_classes"""
    # Sample noise
    # z = FloatTensor(np.random.normal(0, 1, (n_row ** 2, args.latent_dim)))
    # Get labels ranging from 0 to num_classes for n rows
    labels = np.array([num for _ in range(n_row) for num in range(n_row)])

    z = np.zeros((len(labels), args.nz))
    noise = np.random.normal(0, 1, (len(labels), args.nz- args.num_classes))

    class_onehot = np.zeros((len(labels), args.num_classes))
    class_onehot[np.arange(len(labels)), labels] = 1
    z[np.arange(len(labels)), :args.num_classes] = class_onehot[np.arange(len(labels))]
    z[np.arange(len(labels)), args.num_classes:] = noise[np.arange(len(labels)) ]

    z = torch.tensor(z).float().to(device)
    gen_imgs = generator(z)

    out_im = os.path.join(outdir, '{}.png'.format(batches_done))
    save_image(gen_imgs.data, out_im, nrow=n_row, normalize=True)

def cov(x, rowvar=False, bias=False, ddof=None, aweights=None):
    """Estimates covariance matrix like numpy.cov"""
    # ensure at least 2D
    if x.dim() == 1:
        x = x.view(-1, 1)

    # treat each column as a data point, each row as a variable
    if rowvar and x.shape[0] != 1:
        x = x.t()

    if ddof is None:
        if bias == 0:
            ddof = 1
        else:
            ddof = 0

    w = aweights
    if w is not None:
        if not torch.is_tensor(w):
            w = torch.tensor(w, dtype=torch.float)
        w_sum = torch.sum(w)
        avg = torch.sum(x * (w/w_sum)[:,None], 0)
    else:
        avg = torch.mean(x, 0)

    # Determine the normalization
    if w is None:
        fact = x.shape[0] - ddof
    elif ddof == 0:
        fact = w_sum
    elif aweights is None:
        fact = w_sum - ddof
    else:
        fact = w_sum - ddof * torch.sum(w * w) / w_sum

    xm = x.sub(avg.expand_as(x))

    if w is None:
        X_T = xm.t()
    else:
        X_T = torch.mm(torch.diag(w), xm).t()

    c = torch.mm(X_T, xm)
    c = c / fact

    return c.squeeze()


def update_Stats(model, dataloader, num_classes = 10):

    minority_class = [1]
    #C, Q

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
            if not class_id in minority_class:
                for i, (imgs, labels) in enumerate(dataloader):
                    relevant_idx = (labels == class_id)
                    relevant_images = imgs[relevant_idx]
                    relevant_images = relevant_images.to(device)
                    batch_size = relevant_images.shape[0]
                    if batch_size == 0:
                        continue

                    relevant_feats = model(relevant_images)
                    relevant_feats = relevant_feats.view(batch_size, -1)

                    feat_sub_classCenter = torch.sub(relevant_feats, C[class_id]).unsqueeze(2)
                    feat_sub_classCenter_T = feat_sub_classCenter.permute(0,2,1)
                    v = torch.matmul(feat_sub_classCenter, feat_sub_classCenter_T)
                    v_sum = torch.sum(v, dim=0)

                    V += v_sum


    e, v = torch.eig(cov(V), eigenvectors=True)
    Q = v[:150]
    # torch.save(Q, 'Q.pt')
    # torch.save(C, 'C.pt')

    return Q, C




if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--save_dir", type = str, default = 'images_pure/' )
    parser.add_argument("--data_root", type = str, default = 'data/')
    parser.add_argument("--sample_interval", type = int, default = 400)
    parser.add_argument("--image_size", type = int, default = 32)
    parser.add_argument('--nz', type=int, default=110, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--ngpu', type = int, default = 1)
    parser.add_argument('--num_classes', type = int, default = 10)
    #model hyperparameters
    #optimizer hyperparameters
    parser.add_argument("--lr", type = float, default = 0.0002)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument("--num_epochs", type = int, default = 500)
    parser.add_argument("--num_workers", type = int, default = 8)
    #training helpers
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    train_transform = transforms.Compose([
                    transforms.Resize(args.image_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ])

    trainset_pure = cifar10_mixup.Pure_CIFAR10(root=args.data_root,
                        train=True,
                        transform=train_transform,
                        target_transform = None,
                        minority_class = 'automobile',
                        keep_ratio = 0.1)

    # trainset_pure = torchvision.datasets.CIFAR10(root="./data", download=True,
    #                        transform=transforms.Compose([
    #                            transforms.Resize(32),
    #                            transforms.ToTensor(),
    #                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #                        ]))

    trainloader_pure = torch.utils.data.DataLoader(trainset_pure, batch_size=args.batch_size,
                                              shuffle=True, num_workers=args.num_workers)


    generator = networks._netG_CIFAR10(args.ngpu, args.nz, nc = 3).to(device)
    discriminator_backbone = networks._netD_CIFAR10_Backbone(args.ngpu).to(device)
    discriminator_head = networks._netD_CIFAR10_Head(10, 1).to(device)

    dis_criterion = torch.nn.BCELoss()
    aux_criterion = torch.nn.CrossEntropyLoss()

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(itertools.chain(
                                        discriminator_backbone.parameters(),
                                        discriminator_head.parameters()
                                    ),
                                    lr=args.lr, betas=(0.5, 0.999))

    for epoch_iter in range(args.num_epochs):

        Q, C = update_Stats(discriminator_backbone, trainloader_pure)


        for iter, pure_batch in enumerate ( trainloader_pure):


            generator, discriminator_backbone, discriminator_head, d_loss, g_loss, accuracy = training_utils.train_transfer_gan(generator,
                            discriminator_backbone, discriminator_head, pure_batch, device, args,
                            optimizer_G, optimizer_D, dis_criterion, aux_criterion, Q, C)


            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [Real Accuracy: %f]"
                % (epoch_iter+1, args.num_epochs, iter, len(trainloader_pure), d_loss.item(), g_loss.item(), accuracy)
            )

            batches_done = epoch_iter * len(trainloader_pure) + iter
            if batches_done % args.sample_interval == 0:

                sample_image(
                    generator = generator,
                    n_row=args.num_classes,
                    batches_done=batches_done,
                    outdir = args.save_dir,
                    device = device,
                    args = args)
