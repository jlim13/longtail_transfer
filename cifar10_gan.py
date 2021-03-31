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
import itertools

from datasets import cifar10, mnist
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



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--save_dir", type = str, default = 'outputs/transfer_imbalance_gan/' )
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
    parser.add_argument("--num_epochs", type = int, default = 1000)
    parser.add_argument("--num_workers", type = int, default = 8)
    #training helpers
    args = parser.parse_args()

    minority_class_labels = [0, 1]

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    train_transform = transforms.Compose([
                    transforms.Resize(args.image_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ])

    trainset_pure = cifar10.CIFAR10(root='./data',
                        train=True,
                        download=True,
                        transform=train_transform,
                        minority_classes = minority_class_labels,
                        keep_ratio = 0.1)

    # trainset_pure = mnist.MNIST(root = 'data/MNIST', train = True,
    #                      download=True,
    #                      transform=transforms.Compose([
    #                         transforms.ToTensor(),
    #                         transforms.Normalize((0.5,), (0.5,))]),
    #                     minority_classes = minority_class_labels,
    #                     keep_ratio = 0.01
    #                    )

    trainloader_pure = torch.utils.data.DataLoader(trainset_pure, batch_size=args.batch_size,
                                              shuffle=True, num_workers=args.num_workers)

    testset = cifar10.CIFAR10(root=args.data_root,
                                        train=False,
                                        download=True,
                                        transform=train_transform,
                                        minority_classes = None,
                                        keep_ratio = None)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=args.num_workers)

    generator = networks._netG_CIFAR10(args.ngpu, args.nz, nc = 3).to(device)

    discriminator = networks._netD_CIFAR10(args.ngpu).to(device)
    discriminator_backbone = networks._netD_CIFAR10_Backbone(args.ngpu, nc = 3).to(device)
    discriminator_head = networks._netD_CIFAR10_Head(10, 1).to(device)

    dis_criterion = torch.nn.BCELoss()
    aux_criterion = torch.nn.CrossEntropyLoss()

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(itertools.chain(
                                        discriminator_backbone.parameters(),
                                        discriminator_head.parameters()
                                    ),
                                    lr=args.lr, betas=(0.5, 0.999))
    #

    for epoch_iter in range(args.num_epochs):

        if epoch_iter > 20:
            Q, C = training_utils.update_Stats(discriminator_backbone, trainloader_pure, minority_class_labels, device, num_classes = 10, num_features = 1024)
            training_utils.view_centers(C, discriminator_backbone, testloader, device, num_classes = 10, fname = 'class_centers_{}.png'.format(epoch_iter))

        for iter, pure_batch in enumerate ( trainloader_pure):

            if epoch_iter > 20:
                generator, discriminator_backbone, discriminator_head, d_loss, g_loss, accuracy = \
                            training_utils.train_transfer_gan(generator,
                                discriminator_backbone, discriminator_head, pure_batch, device, args,
                                optimizer_G, optimizer_D, dis_criterion, aux_criterion, Q, C, minority_class_labels)
            else:
                generator, discriminator_backbone, discriminator_head, d_loss, g_loss, accuracy = \
                            training_utils.train_regular_gan(generator,
                                discriminator_backbone, discriminator_head, pure_batch, device, args,
                                optimizer_G, optimizer_D, dis_criterion, aux_criterion)
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [Real Accuracy: %f]"
                % (epoch_iter+1, args.num_epochs, iter, len(trainloader_pure), d_loss.item(), g_loss.item(), accuracy)
            )

            batches_done = epoch_iter * len(trainloader_pure) + iter
            if batches_done % args.sample_interval == 0:

                training_utils.sample_image(
                    generator = generator,
                    n_row=args.num_classes,
                    batches_done=batches_done,
                    outdir = args.save_dir,
                    device = device,
                    args = args)
