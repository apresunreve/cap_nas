'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.
'''

import os, sys
from collections import Counter
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import PIL
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import math

from . import autoaugment

_IMAGENET_PCA = {
    'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
    'eigvec': torch.Tensor([
        [-0.5675, 0.7192, 0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948, 0.4203],
    ])
}
lighting_param = 0.1


# Dataset locations are machine-specific, so this package ships NO default
# paths. Each dataset's root comes from an environment variable; resolution is
# lazy, so an unset variable only raises when that dataset is actually used.


def _dataset_root(*env_vars: str) -> str:
    """First env var that is set wins; raise naming all of them if none is."""
    for env_var in env_vars:
        value = os.environ.get(env_var)
        if value:
            return os.path.expanduser(value)
    names = " or ".join(env_vars)
    raise RuntimeError(
        f"Dataset root not configured: set {names} to the directory "
        f"containing this dataset."
    )


class _Root:
    """Placeholder for <env_var>/<subdirs>, resolved on params_dict lookup.

    Several env vars may be given; the first one set wins.
    """

    __slots__ = ("env_vars", "subdirs")

    def __init__(self, env_vars, *subdirs: str):
        self.env_vars = (env_vars,) if isinstance(env_vars, str) else tuple(env_vars)
        self.subdirs = subdirs

    @property
    def env_var(self) -> str:
        return self.env_vars[0]

    def resolve(self) -> str:
        return os.path.join(_dataset_root(*self.env_vars), *self.subdirs)


class _ParamsDict(dict):
    """params_dict whose path entries resolve on access, not at import."""

    def __getitem__(self, key):
        entry = super().__getitem__(key)
        if not isinstance(entry, dict):
            return entry
        return {
            k: (v.resolve() if isinstance(v, _Root) else v)
            for k, v in entry.items()
        }


def _imagenet100_root() -> str:
    return _dataset_root("CAP_NAS_IMAGENET100_ROOT")


_SPLIT_DIRS = ("train", "val", "test")


def _resolve_sample_path(fpath: str, local_root: str) -> str:
    """Resolve a sample path against the configured dataset root.

    Follows symlinks, then re-roots the result under local_root if the target
    does not exist locally, recovering the <split>/<class>/<file> suffix from
    the path itself.
    """
    if os.path.islink(fpath):
        target = os.readlink(fpath)
        if not os.path.isabs(target):
            target = os.path.normpath(os.path.join(os.path.dirname(fpath), target))
        fpath = target
    if os.path.exists(fpath):
        return fpath
    parts = fpath.split(os.sep)
    for split in _SPLIT_DIRS:
        if split in parts:
            idx = len(parts) - 1 - parts[::-1].index(split)
            return os.path.join(local_root.rstrip(os.sep), *parts[idx:])
    return fpath


def _remap_imagenet100_samples(the_dataset, local_root: str) -> None:
    remapped = [
        (_resolve_sample_path(path, local_root), target)
        for path, target in the_dataset.samples
    ]
    the_dataset.samples = remapped
    the_dataset.imgs = remapped

params_dict = _ParamsDict({
    'cifar100': {
        'train_dir': _Root('CAP_NAS_CIFAR100_ROOT'),
        'val_dir': _Root('CAP_NAS_CIFAR100_ROOT'),
        'num_train_samples': 50000,
        'num_val_samples': 10000,
        'num_classes': 100,
    },
    # iNat 2019. Class dirs are 4-digit zero-padded so that ImageFolder's
    # sorted() class order matches the numeric class id.
    'inat': {
        'train_dir': _Root(('CAP_NAS_INAT_ROOT', 'CAP_NAS_INAT_RESIZED_ROOT'), 'train'),
        'val_dir':   _Root(('CAP_NAS_INAT_ROOT', 'CAP_NAS_INAT_RESIZED_ROOT'), 'val'),
        'test_dir':  _Root(('CAP_NAS_INAT_ROOT', 'CAP_NAS_INAT_RESIZED_ROOT'), 'test'),
        'num_train_samples': 262183,
        'num_val_samples': 3030,
        'num_test_samples': 3030,
        'num_classes': 1010,
    },
    'inat_resized': {
        'train_dir': _Root(('CAP_NAS_INAT_ROOT', 'CAP_NAS_INAT_RESIZED_ROOT'), 'train'),
        'val_dir':   _Root(('CAP_NAS_INAT_ROOT', 'CAP_NAS_INAT_RESIZED_ROOT'), 'val'),
        'test_dir':  _Root(('CAP_NAS_INAT_ROOT', 'CAP_NAS_INAT_RESIZED_ROOT'), 'test'),
        'num_train_samples': 262183,
        'num_val_samples': 3030,
        'num_test_samples': 3030,
        'num_classes': 1010,
    },
    'imagenet100':{
        'train_dir': _Root('CAP_NAS_IMAGENET100_ROOT', 'train'),
        'val_dir': _Root('CAP_NAS_IMAGENET100_ROOT', 'val'),
        'test_dir': _Root('CAP_NAS_IMAGENET100_ROOT', 'test'),
        'num_train_samples': 117000,
        'num_val_samples': 5000,
        'num_test_samples': 13000,
        'num_classes': 100,
    },
})


class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone() \
            .mul(alpha.view(1, 3).expand(3, 3)) \
            .mul(self.eigval.view(1, 3).expand(3, 3)) \
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))


def _dataset_targets(dataset):
    """Per-sample class labels, unwrapping Subset wrappers."""
    if isinstance(dataset, torch.utils.data.Subset):
        inner = _dataset_targets(dataset.dataset)
        return [inner[i] for i in dataset.indices]
    if hasattr(dataset, 'targets'):
        return list(dataset.targets)
    return [label for _, label in dataset.samples]


class ClassBalancedSampler(torch.utils.data.WeightedRandomSampler):
    """Sample uniformly over classes, with replacement.

    Draws a uniformly random class, then a random image within it, so a
    long-tailed dataset is balanced over each epoch.

    Provides set_epoch() so it can be used where a DistributedSampler is
    expected. Under DDP each rank draws num_samples // world_size; sampling is
    with replacement, so the indices need no partitioning.
    """

    def __init__(self, dataset, num_samples=None, rank=0, world_size=1, seed=0):
        targets = _dataset_targets(dataset)
        counts = Counter(targets)
        weights = [1.0 / counts[t] for t in targets]
        total = len(targets) if num_samples is None else int(num_samples)
        per_rank = max(1, total // max(world_size, 1))
        generator = torch.Generator()
        generator.manual_seed(seed + rank)
        super().__init__(weights, num_samples=per_rank, replacement=True,
                         generator=generator)
        self._seed = seed
        self._rank = rank
        self.num_classes = len(counts)

    def set_epoch(self, epoch):
        self.generator.manual_seed(self._seed + 1000003 * int(epoch) + self._rank)


def load_imagenet_like(dataset_name, set_name, train_augment, random_erase, auto_augment,
                       data_dir, input_image_size, input_image_crop, rank, world_size,
                       shuffle, batch_size, num_workers, drop_last, dataset_ImageFolderClass,
                       dataloader_testing, frac=1, img_per_class=None, imagenet100_exclude_dir=None,
                       class_balanced=False, balanced_epoch_samples=None):
    resize_image_size = int(math.ceil(input_image_size / input_image_crop))
    transforms_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if dataset_name == 'imagenet100':
        if train_augment == False:
            transform_list = [transforms.ToTensor(), transforms_normalize]
        elif auto_augment:
            transform_list = [ transforms.RandomHorizontalFlip(),
                                autoaugment.ImageNetPolicy(),
                                transforms.ToTensor(),
                                transforms_normalize]
        else:
            transform_list = [transforms.RandomHorizontalFlip(),
                              transforms.ColorJitter(0.4, 0.4, 0.4),
                              transforms.ToTensor(),
                              Lighting(lighting_param, _IMAGENET_PCA['eigval'], _IMAGENET_PCA['eigvec']),
                              transforms_normalize]
    elif dataset_name in ('inat', 'inat_resized'):
        # iNat uses mean=std=0.5 and a squash resize, not ImageNet statistics.
        inat_normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        squash = transforms.Resize((input_image_size, input_image_size),
                                   interpolation=PIL.Image.BICUBIC)
        if train_augment == False:
            transform_list = [squash, transforms.ToTensor(), inat_normalize]
        elif auto_augment:
            transform_list = [squash, transforms.RandomHorizontalFlip(),
                              autoaugment.ImageNetPolicy(),
                              transforms.ToTensor(), inat_normalize]
        else:
            transform_list = [squash, transforms.RandomHorizontalFlip(),
                              transforms.ToTensor(), inat_normalize]
    elif train_augment == False:
        assert random_erase == False and auto_augment == False
        transform_list = [transforms.Resize(resize_image_size, interpolation=PIL.Image.BICUBIC), transforms.CenterCrop(input_image_size),
                          transforms.ToTensor(), transforms_normalize]
        
    else:
        if auto_augment:
            transform_list = [transforms.RandomResizedCrop(input_image_size, interpolation=PIL.Image.BICUBIC),
                              transforms.RandomHorizontalFlip(),
                              autoaugment.ImageNetPolicy(),
                              transforms.ToTensor(),
                              Lighting(lighting_param, _IMAGENET_PCA['eigval'], _IMAGENET_PCA['eigvec']),
                              transforms_normalize]
        else:
            transform_list = [transforms.RandomResizedCrop(input_image_size, interpolation=PIL.Image.BICUBIC),
                              transforms.RandomHorizontalFlip(),
                              transforms.ColorJitter(0.4, 0.4, 0.4),
                              transforms.ToTensor(),
                              Lighting(lighting_param, _IMAGENET_PCA['eigval'], _IMAGENET_PCA['eigvec']),
                              transforms_normalize]
        pass

        if random_erase:
            transform_list.append(transforms.RandomErasing())
    pass

    transformer = transforms.Compose(transform_list)
    the_dataset = dataset_ImageFolderClass(data_dir, transformer)

    if dataset_name == 'imagenet100':
        _remap_imagenet100_samples(the_dataset, _imagenet100_root())

    if dataset_name == 'imagenet100' and set_name == 'train' and imagenet100_exclude_dir is not None:
        exclude_keys = set()
        for class_name in sorted(os.listdir(imagenet100_exclude_dir)):
            class_dir = os.path.join(imagenet100_exclude_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for filename in os.listdir(class_dir):
                exclude_keys.add((class_name, filename))

        keep_samples = []
        removed = 0
        for path, target in the_dataset.samples:
            key = (os.path.basename(os.path.dirname(path)), os.path.basename(path))
            if key in exclude_keys:
                removed += 1
            else:
                keep_samples.append((path, target))
        the_dataset.samples = keep_samples
        the_dataset.imgs = keep_samples
        the_dataset.targets = [target for _, target in keep_samples]
        print(f'[DataLoader] imagenet100 train excluded {removed} validation-overlap images from {imagenet100_exclude_dir}')
    
    if set_name == 'train' and img_per_class is not None and img_per_class != -1:
        print(f'[DataLoader] Using {img_per_class} images per class')
        np.random.seed(42)
        idxs = []
        for class_idx in np.unique(the_dataset.targets):
            targets = torch.tensor(the_dataset.targets)
            indices_of_class_b = (targets == class_idx).nonzero(as_tuple=True)[0]
            print(f'[DataLoader] Using {img_per_class} images for class {class_idx}')
            class_idxs = np.random.choice(indices_of_class_b, img_per_class, replace=True)
            idxs.extend(class_idxs.tolist())
        the_dataset = torch.utils.data.Subset(the_dataset, indices=idxs)
        print(f'[DataLoader] Subsampled dataset to {len(the_dataset)} images')

    elif 'train' in data_dir and frac > 1:
        print('[DataLoader] Shuffling idxs')
        np.random.seed(42)

        idxs = np.random.choice(range(len(the_dataset)),round(len(the_dataset)/frac), replace=False)
        the_dataset = torch.utils.data.Subset(the_dataset, idxs)


    if dataloader_testing:
        tmp_indices = np.arange(0, len(the_dataset))
        kk = 100 if set_name == 'train' else 10
        tmp_indices = np.array_split(tmp_indices, kk)[0]
        the_dataset = torch.utils.data.Subset(the_dataset, indices=tmp_indices)

    if dataset_name == 'imagenet100':
        print(f'[DataLoader] imagenet100 {set_name} data_dir={data_dir}, dataset_size={len(the_dataset)}')

    if class_balanced and set_name == 'train':
        sampler = ClassBalancedSampler(the_dataset,
                                       num_samples=balanced_epoch_samples,
                                       rank=rank, world_size=world_size)
        print(f'[DataLoader] {dataset_name} train: class-balanced sampling over '
              f'{sampler.num_classes} classes, {sampler.num_samples} draws/epoch/rank')
    elif shuffle:
        sampler = torch.utils.data.distributed.DistributedSampler(the_dataset,
                                                                  num_replicas=world_size,
                                                                  rank=rank)
    else:
        sampler = None
        if world_size > 1:
            tmp_indices = np.arange(0, len(the_dataset))
            tmp_indices = np.array_split(tmp_indices, world_size)[rank]
            the_dataset = torch.utils.data.Subset(the_dataset, indices=tmp_indices)

        pass
    pass

    data_loader = torch.utils.data.DataLoader(the_dataset, batch_size=batch_size, shuffle=False,
                                              num_workers=num_workers, pin_memory=True, sampler=sampler,
                                              drop_last=drop_last)

    return {'data_loader': data_loader,
            'sampler': sampler,
            }

def load_cifar_like(dataset_name, set_name, train_augment, random_erase, auto_augment,
                       data_dir, input_image_size, input_image_crop, rank, world_size,
                       shuffle, batch_size, num_workers, drop_last, dataset_ImageFolderClass,
                    dataloader_testing=False, img_per_class=None):

    transforms_normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])

    if train_augment == False:
        assert random_erase == False and auto_augment == False
        if input_image_size > 32:
            transform_list = [transforms.Resize(input_image_size, interpolation=PIL.Image.BICUBIC)]
        else:
            transform_list = []

        transform_list += [transforms.ToTensor(), transforms_normalize]
    else:

        if input_image_size > 32:
            resize_image_size = round(input_image_size / 0.75)
            transform_list = [transforms.Resize(resize_image_size, interpolation=PIL.Image.BICUBIC)]
            transform_list += [transforms.RandomResizedCrop(input_image_size, scale=(0.8, 1.0),
                                                            interpolation=PIL.Image.BICUBIC)]
        else:
            transform_list = [transforms.RandomCrop(input_image_size, padding=4)]

        if auto_augment:
            autoaugment_policy = autoaugment.CIFAR10Policy()
            transform_list += [transforms.RandomHorizontalFlip(), autoaugment_policy,
                               transforms.ToTensor(),
                               transforms_normalize]
        else:
            transform_list += [transforms.RandomHorizontalFlip(),
                               transforms.ToTensor(),
                               transforms_normalize]
        pass
        if random_erase:
            transform_list.append(transforms.RandomErasing())
    pass

    transformer = transforms.Compose(transform_list)

    if dataset_name == 'cifar100':
        the_dataset = datasets.CIFAR100(root=data_dir, train=set_name=='train', download=True, transform=transformer)
    else:
        raise ValueError('Unknown dataset_name=' + dataset_name)

    if img_per_class is not None and img_per_class != -1:
        print(f'[DataLoader] Using {img_per_class} images per class')
        np.random.seed(42)
        idxs = []
        for class_idx in np.unique(the_dataset.targets):
            targets = torch.tensor(the_dataset.targets)
            indices_of_class_b = (targets == class_idx).nonzero(as_tuple=True)[0]
            if len(indices_of_class_b) < img_per_class:
                print(f'[DataLoader] Only {len(indices_of_class_b)} images for class {class_idx}, using all')
                class_idxs = indices_of_class_b
            else:
                print(f'[DataLoader] Using {img_per_class} images for class {class_idx}')
                class_idxs = np.random.choice(indices_of_class_b, img_per_class, replace=False)
            idxs.extend(class_idxs.tolist())
        the_dataset = torch.utils.data.Subset(the_dataset, indices=idxs)
        print(f'[DataLoader] Subsampled dataset to {len(the_dataset)} images')
       
    elif dataloader_testing:
        tmp_indices = np.arange(0, len(the_dataset))
        kk = 100 if set_name == 'train' else 10
        tmp_indices = np.array_split(tmp_indices, kk)[0]
        the_dataset = torch.utils.data.Subset(the_dataset, indices=tmp_indices)

    if shuffle:
        sampler = torch.utils.data.distributed.DistributedSampler(the_dataset,
                                                                  num_replicas=world_size,
                                                                  rank=rank)
    else:
        sampler = None
        if world_size > 1:
            tmp_indices = np.arange(0, len(the_dataset))
            tmp_indices = np.array_split(tmp_indices, world_size)[rank]
            the_dataset = torch.utils.data.Subset(the_dataset, indices=tmp_indices)

        pass
    pass

    data_loader = torch.utils.data.DataLoader(the_dataset, batch_size=batch_size, shuffle=False,
                                              num_workers=num_workers, pin_memory=True, sampler=sampler,
                                              drop_last=drop_last)

    return {'data_loader': data_loader,
            'sampler': sampler,
            }


def _get_data_(dataset_name=None, set_name=None, batch_size=None, train_augment=False, random_erase=False, auto_augment=False,
             input_image_size=224, input_image_crop=0.875, rank=0, world_size=1, shuffle=False,
             num_workers=6, drop_last=False, dataset_ImageFolderClass=None, dataloader_testing=False, argv=None, frac=1, img_per_class=None,
             class_balanced=False, balanced_epoch_samples=None):

    #     pass
    if dataset_name in ['inat', 'inat_resized', 'imagenet100']:
        dataset_params = params_dict[dataset_name]
        if dataset_name == 'imagenet100':
            if set_name == 'train':
                data_dir = dataset_params['train_dir']
                # The test split's images may also be present under train/, so
                # they are excluded from train; val needs no exclusion.
                imagenet100_exclude_dir = dataset_params['test_dir']
            elif set_name == 'test':
                data_dir = dataset_params['test_dir']
                imagenet100_exclude_dir = None
            else:
                data_dir = dataset_params['val_dir']
                imagenet100_exclude_dir = None
        else:
            if set_name == 'train':
                data_dir = dataset_params['train_dir']
            elif set_name == 'test' and 'test_dir' in dataset_params:
                data_dir = dataset_params['test_dir']
            else:
                data_dir = dataset_params['val_dir']
            imagenet100_exclude_dir = None

        if dataset_ImageFolderClass is None:
            dataset_ImageFolderClass = datasets.ImageFolder

        return load_imagenet_like(dataset_name=dataset_name, set_name=set_name, train_augment=train_augment,
                                  random_erase=random_erase, auto_augment=auto_augment,
                                  data_dir=data_dir,
                                  input_image_size=input_image_size, input_image_crop=input_image_crop, rank=rank,
                                  world_size=world_size, shuffle=shuffle, batch_size=batch_size,
                                  num_workers=num_workers, drop_last=drop_last,
                                  dataset_ImageFolderClass=dataset_ImageFolderClass,
                                  dataloader_testing=dataloader_testing,
                                  frac=frac,
                                  img_per_class=img_per_class,
                                  imagenet100_exclude_dir=imagenet100_exclude_dir,
                                  class_balanced=class_balanced,
                                  balanced_epoch_samples=balanced_epoch_samples)

    if dataset_name in ['cifar100']:
        dataset_params = params_dict[dataset_name]
        data_dir = dataset_params['train_dir'] if set_name == 'train' else dataset_params['val_dir']

        if dataset_ImageFolderClass is None:
            dataset_ImageFolderClass = datasets.ImageFolder

        return load_cifar_like(dataset_name=dataset_name, set_name=set_name, train_augment=train_augment,
                               random_erase=random_erase, auto_augment=auto_augment,
                               data_dir=data_dir,
                               input_image_size=input_image_size, input_image_crop=input_image_crop, rank=rank,
                               world_size=world_size, shuffle=shuffle, batch_size=batch_size,
                               num_workers=num_workers, drop_last=drop_last,
                               dataset_ImageFolderClass=dataset_ImageFolderClass,
                               dataloader_testing=dataloader_testing,
                               img_per_class=img_per_class)


def get_data(opt, argv):
    dataset_name = opt.dataset
    batch_size = opt.batch_size_per_gpu
    random_erase = opt.random_erase
    auto_augment = opt.auto_augment
    input_image_size = opt.input_image_size
    input_image_crop = opt.input_image_crop
    rank = opt.rank
    world_size = opt.world_size
    num_workers = opt.workers_per_gpu

    # check if independent training
    if opt.independent_training:
        rank = 0
        world_size = 1


    # load train set
    set_name = 'train'
    if opt.no_data_augment:
        train_augment = False
    else:
        train_augment = True
    shuffle = True
    drop_last = True

    frac = getattr(opt, "frac", 1)

    train_dataset_info = _get_data_(dataset_name, set_name, batch_size, train_augment, random_erase, auto_augment,
             input_image_size, input_image_crop, rank, world_size, shuffle,
             num_workers, drop_last, dataloader_testing=opt.dataloader_testing, argv=argv, frac=frac,
             img_per_class=getattr(opt, "img_per_class", -1),
             class_balanced=getattr(opt, "class_balanced_sampling", False),
             balanced_epoch_samples=getattr(opt, "balanced_epoch_samples", None))

    print('train info got')
    # load val set
    set_name = 'val'
    train_augment = False
    random_erase = False
    auto_augment = False
    shuffle = False
    drop_last = False

    val_dataset_info = _get_data_(dataset_name, set_name, batch_size, train_augment, random_erase, auto_augment,
                                    input_image_size, input_image_crop, rank, world_size, shuffle,
                                    num_workers, drop_last, dataloader_testing=opt.dataloader_testing, argv=argv, frac=1, img_per_class=None)

    # load test set (currently only for inat_resized via JSON)
    set_name = 'test'
    train_augment = False
    random_erase = False
    auto_augment = False
    shuffle = False
    drop_last = False
    if dataset_name in ['inat_resized', 'imagenet100']:
        test_dataset_info = _get_data_(dataset_name, set_name, batch_size, train_augment, random_erase, auto_augment,
                                        input_image_size, input_image_crop, rank, world_size, shuffle,
                                        num_workers, drop_last, dataloader_testing=opt.dataloader_testing, argv=argv)
    else:
        test_dataset_info = val_dataset_info

    return {
        'train_loader' : train_dataset_info['data_loader'],
        'val_loader' : val_dataset_info['data_loader'],
        'test_loader' : test_dataset_info['data_loader'],
        'train_sampler': train_dataset_info['sampler'],
        'val_sampler': val_dataset_info['sampler'],
        'test_sampler': test_dataset_info['sampler'],
    }


