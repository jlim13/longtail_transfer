import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.cm as cm
import os
from torchvision.utils import save_image
from torch.distributions.multivariate_normal import MultivariateNormal


def view_centers(C, model, dataloader, device, num_classes = 397, fname = 'samples.png'):

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

    for class_id, center in enumerate(C):
        center = center.detach().cpu().numpy()
        centers.append(center)
        class_center_ids.append(num_embedded_samples + class_id)
        targets.append(num_embedded_samples + class_id )

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

        plt.scatter(xs, ys, c = colors, alpha = 0.1)

    for idx, class_center_id in enumerate(class_center_ids):
        class_center_embedding = samples_embedded[class_center_id]
        x, y = class_center_embedding

        color = [color_dict[idx]]
        plt.scatter(x, y, c = color, alpha = 1.0)
        plt.annotate('Center_{}'.format(idx), (x,y) )

    plt.savefig(fname)
    plt.clf()




def update_Stats(model, dataloader, minority_class_labels, device, num_classes = 397, num_features = 2048):

    print ("Calculating Statistics")
    #C, Q
    model.eval()

    class_counts = {}
    V = torch.zeros(num_features, num_features).to(device)
    C = torch.zeros(num_classes, num_features).to(device)

    with torch.no_grad():

        for i, (imgs, labels) in enumerate(dataloader):

            labels = labels.to(device)
            imgs = imgs.to(device)
            feats = model(imgs)
            # dummy_tensor = torch.stack([torch.full_like(feats[0], x) for x in labels.detach().cpu().numpy()])
            C.index_add_(0, labels, feats)

            # C[labels.detach().cpu().numpy()] += feats
            # print (dummy_tensor)
            # print (labels.detach().cpu().numpy())

            for label in labels:
                if not label.item() in class_counts:

                    class_counts[label.item()] = 1
                else:
                    class_counts[label.item()] += 1

        for idx, running_sum in enumerate(C):
            C[idx] = running_sum / class_counts[idx]

        num_instances = 0
        for i, (imgs, labels) in enumerate(dataloader):

            imgs = imgs.to(device)

            majority_class_idx = [i for i, elm in enumerate(labels) if not elm in minority_class_labels]
            majority_class_imgs = imgs[majority_class_idx]
            class_centers = C[labels[majority_class_idx].detach().cpu().numpy()]

            majority_feats = model(majority_class_imgs)
            feat_sub_classCenter = majority_feats - class_centers
            feat_sub_classCenter_T = feat_sub_classCenter.permute(1,0)
            v = torch.matmul(feat_sub_classCenter_T, feat_sub_classCenter)
            num_instances += len(majority_class_idx)
            V += v

        V /= num_instances


    e, v = torch.eig(V, eigenvectors=True)
    e = e[:,0]
    sorted_idx = torch.argsort(e, dim=0, descending=True)#[:,0]
    e, v = e[sorted_idx], v[sorted_idx]

    Q = v[:15]
    e = e[:15]

    QQT = torch.matmul(Q.permute(1,0), Q)
    E = torch.diag(e)
    # QQT = torch.matmul(torch.matmul(Q.permute(1,0), E), Q)

    # ones = torch.ones(num_features)
    # QQT = torch.diag(ones).to(device)

    model.train()

    return QQT, C
    # return V, C

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

        transfered_feat = min_class_center + torch.matmul(Q, (maj_feat - maj_class_center).squeeze(0) ).unsqueeze(0)
        # dist = MultivariateNormal(min_class_center, Q)
        # transfered_feat = dist.sample().unsqueeze(0)
        transfered_feats.append(transfered_feat)
        transfered_labels.append(minority_label)


    transfered_feats = torch.cat(transfered_feats, dim=0)
    transfered_labels = torch.tensor(transfered_labels).long().to(device)

    return transfered_feats, transfered_labels


def train_regular_gan(G, D_backbone, D_head, data, device, args, generator_optim, discriminator_optim, bce_criterion, cls_criterion):

    images, labels = data
    images = images.to(device)
    labels = labels.to(device)

    batch_size = images.shape[0]

    valid = torch.tensor(np.ones((batch_size)), requires_grad = False).float().to(device)
    fake = torch.tensor(np.zeros((batch_size)), requires_grad = False).float().to(device)

    # -----------------
    #  Train Generator
    # -----------------

    generator_optim.zero_grad()

    # Sample noise and labels as generator input
    z = np.zeros((batch_size, args.nz))
    noise = np.random.normal(0, 1, (batch_size, args.nz - args.num_classes))
    gen_labels = np.random.randint(0, args.num_classes, batch_size)

    class_onehot = np.zeros((batch_size, args.num_classes))
    class_onehot[np.arange(batch_size), gen_labels] = 1
    z[np.arange(batch_size), :args.num_classes] = class_onehot[np.arange(batch_size)]
    z[np.arange(batch_size), args.num_classes:] = noise[np.arange(batch_size)]

    z = torch.tensor(z).float().to(device)

    # Generate a batch of images/sample from fake
    gen_imgs = G(z)
    gen_feats = D_backbone(gen_imgs)

    validity, pred_label = D_head(gen_feats)

    gen_labels_ = torch.tensor(gen_labels).long().to(device)

    g_loss = bce_criterion(validity, valid) + cls_criterion(pred_label, gen_labels_)
    g_loss.backward()
    generator_optim.step()

    # ---------------------
    #  Train Discriminator
    # ---------------------

    discriminator_optim.zero_grad()

    # Loss for real images
    real_feats = D_backbone(images)
    real_pred, real_aux = D_head(real_feats)

    d_real_loss = 0.5 * (bce_criterion(real_pred, valid) +  cls_criterion(real_aux, labels) )

    # Loss for fake images
    fake_feats = D_backbone(gen_imgs.detach())
    fake_pred, fake_aux = D_head(fake_feats)
    d_fake_loss = 0.5 * (bce_criterion(fake_pred, fake) + cls_criterion(fake_aux, gen_labels_) )

    # Total discriminator loss
    d_loss = d_real_loss + d_fake_loss

    # # Calculate discriminator (aux classifier) accuracy


    # total += targets.size(0)
    # correct += (lam * predicted.eq(targets_a.data).cpu().sum().float()
                # + (1 - lam) * predicted.eq(targets_b.data).cpu().sum().float())
    d_loss.backward()
    discriminator_optim.step()

    real_accuracy = compute_acc(real_aux, labels)

    return G, D_backbone, D_head, d_loss, g_loss, real_accuracy


def compute_acc(preds, labels):
    correct = 0
    preds_ = preds.data.max(1)[1]
    correct = preds_.eq(labels.data).cpu().sum()
    acc = float(correct) / float(len(labels.data))
    return acc


def train_transfer_gan(G, D_backbone, D_head, data, device, args, generator_optim, discriminator_optim, bce_criterion, cls_criterion, Q, C, minority_class_labels):

    images, labels = data
    images = images.to(device)
    labels = labels.to(device)

    batch_size = images.shape[0]

    valid = torch.tensor(np.ones((batch_size)), requires_grad = False).float().to(device)
    fake = torch.tensor(np.zeros((batch_size)), requires_grad = False).float().to(device)

    # -----------------
    #  Train Generator
    # -----------------

    generator_optim.zero_grad()

    # Sample noise and labels as generator input
    z = np.zeros((batch_size, args.nz))
    noise = np.random.normal(0, 1, (batch_size, args.nz - args.num_classes))
    gen_labels = np.random.randint(0, args.num_classes, batch_size)

    class_onehot = np.zeros((batch_size, args.num_classes))
    class_onehot[np.arange(batch_size), gen_labels] = 1
    z[np.arange(batch_size), :args.num_classes] = class_onehot[np.arange(batch_size)]
    z[np.arange(batch_size), args.num_classes:] = noise[np.arange(batch_size)]

    z = torch.tensor(z).float().to(device)

    # Generate a batch of images/sample from fake
    gen_imgs = G(z)
    gen_feats = D_backbone(gen_imgs)

    validity, pred_label = D_head(gen_feats)
    gen_labels_ = torch.tensor(gen_labels).long().to(device)

    g_loss = bce_criterion(validity, valid) + cls_criterion(pred_label, gen_labels_)
    # g_loss = get_entropy_loss(validity) + cls_criterion(pred_label, gen_labels_)
    g_loss.backward()
    generator_optim.step()

    # ---------------------
    #  Train Discriminator
    # ---------------------

    discriminator_optim.zero_grad()

    # Loss for real images

    real_feats = D_backbone(images)
    real_pred, real_aux = D_head(real_feats)
    d_real_loss = (bce_criterion(real_pred, valid) +  cls_criterion(real_aux, labels) )

    transfered_features, transfered_labels = transfer_feats(real_feats, labels, Q, C, minority_class_labels, device = device)
    transfered_pred, transfered_aux = D_head(transfered_features)
    valid_transfer = torch.tensor(np.ones((len(transfered_pred))), requires_grad = False).float().to(device)
    d_transfer_loss = (bce_criterion(transfered_pred, valid_transfer) + cls_criterion(transfered_aux, transfered_labels) )
    # Loss for fake images
    fake_feats = D_backbone(gen_imgs.detach())
    fake_pred, fake_aux = D_head(fake_feats)
    d_fake_loss = (bce_criterion(fake_pred, fake) + cls_criterion(fake_aux, gen_labels_) )

    # Total discriminator loss
    d_loss = (d_real_loss + d_transfer_loss + d_fake_loss) / 3

    d_loss.backward()
    discriminator_optim.step()

    # # Calculate discriminator (aux classifier) accuracy
    real_accuracy = compute_acc(real_aux, labels)


    return G, D_backbone, D_head, d_loss, g_loss, real_accuracy


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
