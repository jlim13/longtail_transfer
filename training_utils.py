import torch
import numpy as np
import random

def mixup_criterion(loss_fn, pred, y_a, y_b, lam, device):

    # loss = (lam * criterion(pred, y_a)) + ((1 - lam) * criterion(pred, y_b))

    y_a = np.where(y_a == 1)[1]
    y_b = np.where(y_b == 1)[1]

    y_a = torch.tensor(y_a).long().to(device)
    y_b = torch.tensor(y_b).long().to(device)


    loss_a = lam * loss_fn(pred, y_a)
    loss_b = (1 - lam) * loss_fn(pred, y_b)

    loss = loss_a + loss_b
    loss_mean = torch.mean(loss)

    return loss_mean


def get_entropy_loss(out):
    return -torch.mean(torch.log(torch.nn.functional.softmax(out + 1e-6, dim=-1)))

def mixup_oneHots(batch_size, y_a, y_b, lam, num_classes):

    gen_labels_A = y_a.detach().cpu().numpy()
    gen_labels_B = y_b.detach().cpu().numpy()

    class_onehot_A = np.zeros((batch_size, num_classes))
    class_onehot_B = np.zeros((batch_size, num_classes))
    class_onehot_A[np.arange(batch_size), gen_labels_A] = 1
    class_onehot_B[np.arange(batch_size), gen_labels_B] = 1

    lam_a = np.expand_dims(lam.detach().cpu().numpy(), 1)
    lam_b = 1-lam_a

    class_onehot_A_mixed = class_onehot_A * lam_a
    class_onehot_B_mixed = class_onehot_B * lam_b

    mixed_onehots = class_onehot_A_mixed + class_onehot_B_mixed
    assert( np.allclose(np.sum(mixed_onehots, axis = 1), 1))

    return mixed_onehots, class_onehot_A, class_onehot_B

#original
'''
def train_mixup_gan(G, D, data, device, args, generator_optim, discriminator_optim, bce_criterion, cls_criterion):

    mixed_image, y_a, y_b, lamy = data

    mixed_image = mixed_image.to(device)
    y_a = y_a.to(device)
    y_b = y_b.to(device)
    lamy = lamy.to(device)


    batch_size = mixed_image.shape[0]

    valid = torch.tensor(np.ones((batch_size )), requires_grad = False).float().to(device)
    fake = torch.tensor(np.zeros((batch_size )), requires_grad = False).float().to(device)

    # -----------------
    #  Train Generator
    # -----------------
    generator_optim.zero_grad()

    # Sample noise and labels as generator input
    z = np.zeros((batch_size, args.nz))
    noise = np.random.normal(0, 1, (batch_size, args.nz - args.num_classes))

    mixed_onehots, onehot_A, onehot_B = mixup_oneHots(batch_size, args.num_classes, lamy)

    z[np.arange(batch_size), :args.num_classes] = mixed_onehots[np.arange(batch_size)]
    z[np.arange(batch_size), args.num_classes:] = noise[np.arange(batch_size)]

    z = torch.tensor(z).float().to(device)

    # Generate a batch of images/sample from fake
    gen_imgs = G(z)
    validity, pred_label = D(gen_imgs)

    g_loss = bce_criterion(validity, valid) + \
        mixup_criterion(cls_criterion, pred_label, onehot_A, onehot_B, lamy, device)

    g_loss.backward()
    generator_optim.step()


    # ---------------------
    #  Train Discriminator
    # ---------------------

    discriminator_optim.zero_grad()

    # Loss for real images
    real_pred, real_aux = D(mixed_image)

    d_real_loss = 0.5 * (bce_criterion(real_pred, valid) + \
        mixup_criterion(cls_criterion, real_aux, onehot_A, onehot_B, lamy, device))

    # Loss for fake images
    fake_pred, fake_aux = D(gen_imgs.detach())
    d_fake_loss = 0.5 * (bce_criterion(fake_pred, fake) +  \
        mixup_criterion(cls_criterion, fake_aux, onehot_A, onehot_B, lamy, device))
    # Total discriminator loss
    d_loss = (d_real_loss + d_fake_loss) / 2

    # # Calculate discriminator (aux classifier) accuracy


    d_loss.backward()
    discriminator_optim.step()

    return G, D
'''


def train_regular_gan(G, D, data, device, args, generator_optim, discriminator_optim, bce_criterion, cls_criterion):

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

    validity, pred_label = D(gen_imgs)
    gen_labels_ = torch.tensor(gen_labels).long().to(device)

    # g_loss = bce_criterion(validity, valid) + cls_criterion(pred_label, gen_labels_)
    g_loss = get_entropy_loss(validity) + cls_criterion(pred_label, gen_labels_)
    g_loss.backward()
    generator_optim.step()

    # ---------------------
    #  Train Discriminator
    # ---------------------

    discriminator_optim.zero_grad()

    # Loss for real images

    real_pred, real_aux = D(images)

    d_real_loss = 0.5 * (bce_criterion(real_pred, valid) +  cls_criterion(real_aux, labels) )

    # Loss for fake images
    fake_pred, fake_aux = D(gen_imgs.detach())
    d_fake_loss = 0.5 * (bce_criterion(fake_pred, fake) + cls_criterion(fake_aux, gen_labels_) )

    # Total discriminator loss
    d_loss = d_real_loss + d_fake_loss

    # # Calculate discriminator (aux classifier) accuracy


    # total += targets.size(0)
    # correct += (lam * predicted.eq(targets_a.data).cpu().sum().float()
                # + (1 - lam) * predicted.eq(targets_b.data).cpu().sum().float())
    d_loss.backward()
    discriminator_optim.step()

    return G, D, d_loss, g_loss


def train_mixup_gan(G, D, data, device, args, generator_optim, discriminator_optim, bce_criterion, cls_criterion):

    mixed_image, imga, imgb, y_a, y_b, lamy = data

    mixed_image = mixed_image.to(device)
    imga = imga.to(device)
    imgb = imgb.to(device)
    y_a = y_a.to(device)
    y_b = y_b.to(device)
    lamy = lamy.to(device)

    batch_size = mixed_image.shape[0]

    valid = torch.tensor(np.ones((batch_size )), requires_grad = False).float().to(device)
    fake = torch.tensor(np.zeros((batch_size )), requires_grad = False).float().to(device)
    # mixed_real = torch.tensor(np.full((batch_size ), 2), requires_grad = False).long().to(device)
    # mixed_fake = torch.tensor(np.full((batch_size ), 3), requires_grad = False).long().to(device)

    # -----------------
    #  Train Generator
    # -----------------
    generator_optim.zero_grad()

    # Sample noise and labels as generator input
    z = np.zeros((batch_size, args.nz))
    noise = np.random.normal(0, 1, (batch_size, args.nz - args.num_classes))

    mixed_onehots, onehot_A, onehot_B = mixup_oneHots(batch_size, y_a, y_b, lamy, args.num_classes)

    z[np.arange(batch_size), :args.num_classes] = onehot_B[np.arange(batch_size)]
    z[np.arange(batch_size), args.num_classes:] = noise[np.arange(batch_size)]

    z = torch.tensor(z).float().to(device)

    # Generate a batch of images/sample from fake
    gen_imgs = G(z)
    validity, pred_label = D(gen_imgs)

    # g_loss = bce_criterion(validity, valid) + \
        # mixup_criterion(cls_criterion, pred_label, onehot_A, onehot_B, lamy, device)
    g_loss = bce_criterion(validity, valid) + \
        torch.mean(cls_criterion(pred_label, y_b))
    g_loss.backward()
    generator_optim.step()

    # ---------------------
    #  Train Discriminator
    # ---------------------

    discriminator_optim.zero_grad()

    alpha = float(np.random.random())
    mixed_image = (imga * alpha) + (gen_imgs.detach() * (1-alpha))

    # # Loss for real images
    real_pred, real_aux = D(imga)

    d_real_loss = bce_criterion(real_pred, valid) + \
        torch.mean(cls_criterion(real_aux, y_a))
    #
    # # Loss for fake images
    # fake_pred, fake_aux = D(gen_imgs.detach())
    # d_fake_loss = 0.5 * (bce_criterion(fake_pred, fake) +  \
    #     mixup_criterion(cls_criterion, fake_aux, onehot_A, onehot_B, lamy, device))
    # # Total discriminator loss
    # d_loss = d_real_loss + d_fake_loss

    # Loss for real images
    mixed_pred, mixed_aux = D(mixed_image)

    # d_real_loss = bce_criterion(mixed_pred, valid*alpha) + \
    #     mixup_criterion(cls_criterion, mixed_aux, onehot_A, onehot_B, alpha, device)
    d_mix_loss = alpha*bce_criterion(mixed_pred, valid) +  (1-alpha) * bce_criterion(mixed_pred, valid) + \
        mixup_criterion(cls_criterion, mixed_aux, onehot_A, onehot_B, alpha, device)

    # Loss for fake images
    fake_pred, fake_aux = D(gen_imgs.detach())
    # d_fake_loss = 0.5 * (bce_criterion(fake_pred, fake*(1-alpha)) +  \
    #     mixup_criterion(cls_criterion, fake_aux, onehot_A, onehot_B, lamy, device))
    # # Total discriminator loss
    d_fake_loss = bce_criterion(fake_pred, fake) + \
        torch.mean(cls_criterion(fake_aux, y_b))
    d_loss = (d_mix_loss + d_fake_loss + d_real_loss) / 3
    # d_loss = d_real_loss
    # # Calculate discriminator (aux classifier) accuracy

    d_loss.backward()
    discriminator_optim.step()

    return G, D, d_loss, g_loss


def train_mixup_unconditional_gan(G, D, data, device, args, generator_optim, discriminator_optim, bce_criterion, cls_criterion):

    # lam = np.random.beta(1.0, 1.0)
    lam = 1.0
    real_image, _ = data
    real_image = real_image.to(device)
    batch_size = real_image.shape[0]

    valid = torch.tensor(np.ones((batch_size)), requires_grad = False).float().to(device)
    fake = torch.tensor(np.zeros((batch_size)), requires_grad = False).float().to(device)

    # -----------------
    #  Train Generator
    # -----------------
    generator_optim.zero_grad()

    # Sample noise and labels as generator input
    z = np.random.normal(0, 1, (batch_size, args.nz))
    z = torch.tensor(z).float().to(device)

    # Generate a batch of images/sample from fake
    gen_imgs = G(z)
    validity, cls_preds = D(gen_imgs)
    g_loss = bce_criterion(validity, valid)
    g_loss.backward()
    generator_optim.step()


    # ---------------------
    #  Train Discriminator
    # ---------------------

    discriminator_optim.zero_grad()

    # # Loss for real images
    # mixed_image = lam*real_image + (1.-lam)*gen_imgs.detach()
    # mixed_pred, _ = D(mixed_image)
    # d_loss_a = bce_criterion(mixed_pred, valid * lam)
    #
    # mixed_image = (1-lam)*real_image + lam*gen_imgs.detach()
    # mixed_pred, _ = D(mixed_image)
    # d_loss_b = bce_criterion(mixed_pred, fake * lam)

    real_pred, _ = D(real_image)
    d_loss_a = bce_criterion(real_pred, valid)

    fake_pred, _ = D(gen_imgs.detach())
    d_loss_b = bce_criterion(fake_pred, fake)

    d_loss = d_loss_a + d_loss_b
    d_loss = d_loss/2

    d_loss.backward()
    discriminator_optim.step()

    return G, D, d_loss, g_loss



def transfer_feats(feats, labels, Q, C, minority_class_label, device):

    #
    minority_labels = [1]

    feature_shape = feats.shape
    batch_size = labels.shape[0]

    minority_class_idx = (labels == minority_class_label)
    majority_class_idx = (labels != minority_class_label)
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

        transfered_feat = min_class_center + torch.matmul(torch.matmul(Q.permute(1,0), Q), maj_feat - maj_class_center).unsqueeze(0)
        transfered_feats.append(transfered_feat)
        transfered_labels.append(minority_label)

    transfered_feats = torch.cat(transfered_feats, dim=0)

    transfered_labels = torch.tensor(transfered_labels).long().to(device)

    return transfered_feats, transfered_labels

def compute_acc(preds, labels):
    correct = 0
    preds_ = preds.data.max(1)[1]
    correct = preds_.eq(labels.data).cpu().sum()
    acc = float(correct) / float(len(labels.data))
    return acc

def train_transfer_gan(G, D_backbone, D_head, data, device, args,
            generator_optim, discriminator_optim, bce_criterion, cls_criterion, Q, C):

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

    validity, pred_label = D_head(D_backbone(gen_imgs))
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

    transfered_features, transfered_labels = transfer_feats(real_feats, labels, Q, C, minority_class_label = 1, device = device)
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
