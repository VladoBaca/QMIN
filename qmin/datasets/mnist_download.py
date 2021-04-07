import torch
from torchvision.datasets import MNIST
from torchvision.datasets.mnist import read_image_file, read_label_file
from torchvision.datasets.utils import extract_archive
from typing import Optional, Callable
import os


def load_and_extract_archive(
        source_root: str,
        target_root: str,
        filename: Optional[str] = None,
        md5: Optional[str] = None,
        remove_finished: bool = False,
) -> None:
    archive = os.path.join(source_root, filename)
    print("Extracting {} to {}".format(archive, target_root))
    extract_archive(archive, target_root, remove_finished)


class MNIST_local(MNIST):
    """

    folder :
    which contains files below:
        train-images-idx3-ubyte.gz
        train-labels-idx1-ubyte.gz
        t10k-images-idx3-ubyte.gz
        t10k-labels-idx1-ubyte.gz

    root:
    the same as MNIST.root

    """

    def __init__(
            self,
            folder: str,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:

        super(MNIST, self).__init__(root, transform=transform,
                                    target_transform=target_transform)
        self.train = train  # training set or test set

        self.load(folder)

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file
        self.data, self.targets = torch.load(os.path.join(self.processed_folder, data_file))

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, MNIST.__name__, 'raw')

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, MNIST.__name__, 'processed')

    def load(self, folder):
        if self._check_exists():
            return
        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)

        for url, md5 in self.resources:
            filename = url.rpartition('/')[2]
            load_and_extract_archive(source_root=folder, target_root=self.raw_folder, filename=filename,
                                     md5=md5)  # NOTICE

        # process and save as torch files
        print('Processing...')

        training_set = (
            read_image_file(os.path.join(self.raw_folder, 'train-images-idx3-ubyte')),
            read_label_file(os.path.join(self.raw_folder, 'train-labels-idx1-ubyte'))
        )
        test_set = (
            read_image_file(os.path.join(self.raw_folder, 't10k-images-idx3-ubyte')),
            read_label_file(os.path.join(self.raw_folder, 't10k-labels-idx1-ubyte'))
        )
        with open(os.path.join(self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')