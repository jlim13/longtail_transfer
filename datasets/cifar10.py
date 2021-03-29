from PIL import Image
import os
import os.path
import numpy as np
import pickle
from typing import Any, Callable, Optional, Tuple
import torchvision.transforms as transforms
import torchvision
import torch
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import download_url, download_and_extract_archive, extract_archive, \
    verify_str_arg,  check_integrity

# classes = ('plane', 'car', 'bird', 'cat',
           # 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#{'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}


class CIFAR10(VisionDataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            minority_classes = None,
            keep_ratio: float = None
    ) -> None:

        super(CIFAR10, self).__init__(root, transform=transform,
                                      target_transform=target_transform)

        self.train = train  # training set or test set
        self.minority_classes = minority_classes
        self.keep_ratio = keep_ratio

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data: Any = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)

            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        self._load_meta()

        if self.minority_classes and self.keep_ratio:
            self._truncate_data()

    def _truncate_data(self) -> None:

        self.idx_to_class = {v:k for k,v in self.class_to_idx.items()}

        freq_dict = {}
        for idx, label in enumerate(self.targets):
            if label in freq_dict:
                freq_dict[label] += 1
            else:
                freq_dict[label] = 1

        n_sample_dict = {}
        minority_class_counter_dict = {x: 0 for x in self.minority_classes}

        for minority_class in self.minority_classes:
            n_samples = int(freq_dict[minority_class] * self.keep_ratio)
            n_sample_dict[minority_class] = n_samples
        #
        self.truncated_data = []
        self.truncated_labels = []
        self.class_to_dataIdxs = {x: [] for x in np.unique(self.targets)}

        for idx, (datapoint, label) in enumerate(zip(self.data, self.targets)):
            if label in self.minority_classes:

                if minority_class_counter_dict[label] >= n_sample_dict[label]:
                    continue
                else:
                    self.truncated_data.append(datapoint)
                    self.truncated_labels.append(label)
                    minority_class_counter_dict[label] += 1
                    self.class_to_dataIdxs[label].append(len(self.truncated_data)-1)

            else:
                self.truncated_data.append(datapoint)
                self.truncated_labels.append(label)
                self.class_to_dataIdxs[label].append(len(self.truncated_data)-1)


        self.data = self.truncated_data
        self.targets = self.truncated_labels
        # count = 0
        # for class_label, idxs in self.class_to_dataIdxs.items():
        #     inner_count = 0
        #     for idx in idxs:
        #         image, targ = self.data[idx], self.targets[idx]
        #
        #         targ_name = self.idx_to_class[targ]
        #         im_name = '{}_{}.png'.format(targ_name, idx)
        #         im = Image.fromarray(image)
        #         im.save(im_name)
        #         count += 1
        #         inner_count += 1
        #
        #         if inner_count == 5:
        #             break

    def _load_meta(self) -> None:
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img_pil = Image.fromarray(img)

        if self.transform is not None:
            img_transform = self.transform(img_pil)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img_transform,  target

    def __len__(self) -> int:
        return len(self.data)

    def _check_integrity(self) -> bool:
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self) -> None:
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def extra_repr(self) -> str:
        return "Split: {}".format("Train" if self.train is True else "Test")


class CIFAR100(CIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }



# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#
#
# trainset = CIFAR10(root='./data',
#                     train=True,
#                     download=True,
#                     transform=transform,
#                     minority_class = 'automobile',
#                     keep_ratio = 0.2)
#
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
#                                           shuffle=True, num_workers=2)
