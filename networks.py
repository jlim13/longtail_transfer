import torch
import torch.nn as nn


class _netG_CIFAR10(nn.Module):
    def __init__(self, ngpu, nz, nc):
        super(_netG_CIFAR10, self).__init__()
        self.ngpu = ngpu
        self.nz = nz

        # first linear layer
        self.fc1 = nn.Linear(self.nz, 384)
        # Transposed Convolution 2
        self.tconv2 = nn.Sequential(
            nn.ConvTranspose2d(384, 192, 4, 1, 0, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
        )
        # Transposed Convolution 3
        self.tconv3 = nn.Sequential(
            nn.ConvTranspose2d(192, 96, 4, 2, 1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(True),
        )
        # Transposed Convolution 4
        self.tconv4 = nn.Sequential(
            nn.ConvTranspose2d(96, 48, 4, 2, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(True),
        )
        # Transposed Convolution 4
        self.tconv5 = nn.Sequential(
            nn.ConvTranspose2d(48, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            input = input.view(-1, self.nz)
            fc1 = nn.parallel.data_parallel(self.fc1, input, range(self.ngpu))
            fc1 = fc1.view(-1, 384, 1, 1)
            tconv2 = nn.parallel.data_parallel(self.tconv2, fc1, range(self.ngpu))
            tconv3 = nn.parallel.data_parallel(self.tconv3, tconv2, range(self.ngpu))
            tconv4 = nn.parallel.data_parallel(self.tconv4, tconv3, range(self.ngpu))
            tconv5 = nn.parallel.data_parallel(self.tconv5, tconv4, range(self.ngpu))
            output = tconv5
        else:
            input = input.view(-1, self.nz)
            fc1 = self.fc1(input)
            fc1 = fc1.view(-1, 384, 1, 1)
            tconv2 = self.tconv2(fc1)
            tconv3 = self.tconv3(tconv2)
            tconv4 = self.tconv4(tconv3)
            tconv5 = self.tconv5(tconv4)
            output = tconv5
        return output


class _netD_CIFAR10(nn.Module):
    def __init__(self, ngpu, num_classes=10, num_discrim_classes = 1, nc = 3):
        super(_netD_CIFAR10, self).__init__()
        self.ngpu = ngpu

        # Convolution 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(nc, 16, 3, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 4
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 5
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 6
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # discriminator fc
        self.fc_dis = nn.Linear(4*4*512, num_discrim_classes)
        # aux-classifier fc
        self.fc_aux = nn.Linear(4*4*512, num_classes)
        # softmax and sigmoid
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            conv1 = nn.parallel.data_parallel(self.conv1, input, range(self.ngpu))
            conv2 = nn.parallel.data_parallel(self.conv2, conv1, range(self.ngpu))
            conv3 = nn.parallel.data_parallel(self.conv3, conv2, range(self.ngpu))
            conv4 = nn.parallel.data_parallel(self.conv4, conv3, range(self.ngpu))
            conv5 = nn.parallel.data_parallel(self.conv5, conv4, range(self.ngpu))
            conv6 = nn.parallel.data_parallel(self.conv6, conv5, range(self.ngpu))
            flat6 = conv6.view(-1, 4*4*512)
            fc_dis = nn.parallel.data_parallel(self.fc_dis, flat6, range(self.ngpu))
            fc_aux = nn.parallel.data_parallel(self.fc_aux, flat6, range(self.ngpu))
        else:

            conv1 = self.conv1(input)
            conv2 = self.conv2(conv1)
            conv3 = self.conv3(conv2)
            conv4 = self.conv4(conv3)
            conv5 = self.conv5(conv4)
            conv6 = self.conv6(conv5)
            flat6 = conv6.view(-1, 4*4*512)
            fc_dis = self.fc_dis(flat6)
            fc_aux = self.fc_aux(flat6)

        realfake = self.sigmoid(fc_dis).view(-1, 1).squeeze(1)
        return realfake , fc_aux

class _netD_CIFAR10_Backbone(nn.Module):
    def __init__(self, ngpu, nc = 3):
        super(_netD_CIFAR10_Backbone, self).__init__()
        self.ngpu = ngpu

        # Convolution 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(nc, 16, 3, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 4
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 5
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 6
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        self.linear_layer = nn.Linear(4*4*512, 1024)
        # discriminator fc
        # self.fc_dis = nn.Linear(4*4*512, num_discrim_classes)
        # # aux-classifier fc
        # self.fc_aux = nn.Linear(4*4*512, num_classes)
        # # softmax and sigmoid
        # self.softmax = nn.Softmax()
        # self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            conv1 = nn.parallel.data_parallel(self.conv1, input, range(self.ngpu))
            conv2 = nn.parallel.data_parallel(self.conv2, conv1, range(self.ngpu))
            conv3 = nn.parallel.data_parallel(self.conv3, conv2, range(self.ngpu))
            conv4 = nn.parallel.data_parallel(self.conv4, conv3, range(self.ngpu))
            conv5 = nn.parallel.data_parallel(self.conv5, conv4, range(self.ngpu))
            conv6 = nn.parallel.data_parallel(self.conv6, conv5, range(self.ngpu))
            flat6 = conv6.view(-1, 4*4*512)
            fc_dis = nn.parallel.data_parallel(self.fc_dis, flat6, range(self.ngpu))
            fc_aux = nn.parallel.data_parallel(self.fc_aux, flat6, range(self.ngpu))
        else:

            conv1 = self.conv1(input)
            conv2 = self.conv2(conv1)
            conv3 = self.conv3(conv2)
            conv4 = self.conv4(conv3)
            conv5 = self.conv5(conv4)
            conv6 = self.conv6(conv5)
            flat6 = conv6.view(-1, 4*4*512)

        return self.linear_layer(flat6)

class _netD_CIFAR10_Head(nn.Module):
    def __init__(self,  num_classes=10, num_discrim_classes = 1):
        super(_netD_CIFAR10_Head, self).__init__()

        # self.linear_layer = nn.Linear(4*4*512, 1024)
        # discriminator fc
        self.fc_dis = nn.Linear(1024, num_discrim_classes)
        # aux-classifier fc
        self.fc_aux = nn.Linear(1024, num_classes)
        # softmax and sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):

        fc_dis = self.fc_dis(input)
        fc_aux = self.fc_aux(input)

        realfake = self.sigmoid(fc_dis).view(-1, 1).squeeze(1)

        return realfake , fc_aux


class Generator_MNIST(nn.Module):
    def __init__(self, nz, out_channel):
        super(Generator_MNIST, self).__init__()
        self.nz = nz
        self.out_channel = out_channel
        ngf = 28

        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( self.nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, self.out_channel, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):

        input = input.view(-1, self.nz, 1, 1)

        return self.main(input)


class Discriminator_Backbone_MNIST(nn.Module):
    def __init__(self, num_channels):
        super(Discriminator_Backbone_MNIST, self).__init__()
        self.num_channels = num_channels
        ndf = 28

        # input is (nc) x 64 x 64
        self.conv1 = nn.Conv2d(num_channels, ndf, 4, 2, 1, bias=False)
        self.lr1 = nn.LeakyReLU(0.2, inplace=True)
        # state size. (ndf) x 32 x 32
        self.conv2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(ndf * 2)
        self.lr2 = nn.LeakyReLU(0.2, inplace=True)
        # state size. (ndf*2) x 16 x 16
        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)
        self.bn3  =  nn.BatchNorm2d(ndf * 4)
        self.lr3 = nn.LeakyReLU(0.2, inplace=True)
        # state size. (ndf*4) x 8 x 8
        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(ndf * 8)
        self.lr4 = nn.LeakyReLU(0.2, inplace=True)
        # state size. (ndf*8) x 4 x 4
        self.conv5 = nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)
        self.out = nn.Sigmoid()

        self.fc_layer = nn.Linear(3584, 1024)


    def forward(self, input):

        batch_size = input.shape[0]
        out1 = self.lr1(self.conv1(input))
        out2 = self.lr2(self.bn2(self.conv2(out1)))
        out3 = self.lr3(self.bn3(self.conv3(out2)))
        out4 = self.lr4(self.bn4(self.conv4(out3)))

        feats = out4.view(batch_size, -1)
        feats = self.fc_layer(feats)

        return feats

class Discriminator_Head_MNIST(nn.Module):
    def __init__(self, num_classes):
        super(Discriminator_Head_MNIST, self).__init__()
        self.num_classes = num_classes
        ndf = 28
        self.cls = nn.Linear(1024, self.num_classes)
        self.valid = nn.Linear(1024, 1)
        self.out = nn.Sigmoid()

    def forward(self, input):

        batch_size = input.shape[0]
        # feats = input.view(batch_size, -1)
        cls_logits = self.cls(input)

        validity = self.out(self.valid(input)).squeeze(1)
        
        return validity, cls_logits

class Discriminator_MNIST(nn.Module):
    def __init__(self, num_channels, num_classes):
        super(Discriminator_MNIST, self).__init__()
        self.num_channels = num_channels
        self.num_classes = num_classes
        ndf = 28

        # input is (nc) x 64 x 64
        self.conv1 = nn.Conv2d(num_channels, ndf, 4, 2, 1, bias=False)
        self.lr1 = nn.LeakyReLU(0.2, inplace=True)
        # state size. (ndf) x 32 x 32
        self.conv2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(ndf * 2)
        self.lr2 = nn.LeakyReLU(0.2, inplace=True)
        # state size. (ndf*2) x 16 x 16
        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)
        self.bn3  =  nn.BatchNorm2d(ndf * 4)
        self.lr3 = nn.LeakyReLU(0.2, inplace=True)
        # state size. (ndf*4) x 8 x 8
        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(ndf * 8)
        self.lr4 = nn.LeakyReLU(0.2, inplace=True)
        # state size. (ndf*8) x 4 x 4
        self.conv5 = nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)
        self.out = nn.Sigmoid()

        self.cls = nn.Linear(3584, self.num_classes)


    def forward(self, input):

        batch_size = input.shape[0]
        out1 = self.lr1(self.conv1(input))
        out2 = self.lr2(self.bn2(self.conv2(out1)))
        out3 = self.lr3(self.bn3(self.conv3(out2)))
        out4 = self.lr4(self.bn4(self.conv4(out3)))

        cls_feats = out4.view(batch_size, -1)
        cls_logits = self.cls(cls_feats)

        out5 = self.conv5(out4)

        validity = self.out(out5).view(-1)

        return validity, cls_logits
