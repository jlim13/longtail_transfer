from torch.utils.data import Dataset, DataLoader
import torch
import os
import torchvision
from PIL import Image
import numpy as np
import warnings
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

class SunDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data_root, data_txt_file, instance_threshold = 100, transform=None):

        self.data_root = data_root
        self.data_txt_file = data_txt_file
        self.transform = transform
        self.data_points = []
        self.instance_threshold = instance_threshold #if below this, then it is a tail class
        self.freq_dict = {}
        self.tail_classes = []
        self.head_classes = []

        with open(self.data_txt_file) as f:
            for line in f:
                line = line.strip ('\n')
                img_fp, label = line.split(' ')
                img_fp = img_fp[1:]
                img_fp = os.path.join(self.data_root, img_fp, )

                label = int(label)-1
                self.data_points.append([img_fp, label])

                if not label in self.freq_dict:
                    self.freq_dict[label] = 1
                else:
                    self.freq_dict[label] += 1

        for class_id, freq in self.freq_dict.items():
            if freq < self.instance_threshold:
                self.tail_classes.append(class_id)
            else:
                self.head_classes.append(class_id)



    def __len__(self):
        return len(self.data_points)

    def __getitem__(self, idx):

        img_fp, label = self.data_points[idx]
        with open(img_fp, 'rb') as f:
            img = Image.open(f).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, label




if __name__ == '__main__':

    train_transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(224),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    dataset = SunDataset(data_root = '/data/', data_txt_file = '/data/sun397_train_lt.txt', transform = train_transform)
    dataloader = DataLoader(dataset, batch_size = 128, shuffle = True, num_workers = 0)
    print (len(dataset))

    for i, x in enumerate(dataloader):
        img, label = x
        print (img.shape, i)
