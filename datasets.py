# datasets.py (Corrected)
import os
import torch
import numpy as np
import pandas as pd
import torch.utils.data as data

# from .CIFAR_M import CIFAR_M  # OLD - Relative import
# from .dataset_folder import ImageFolder # OLD
from CIFAR_M import CIFAR_M  # NEW - Absolute import
from dataset_folder import ImageFolder # NEW
from timm.data import create_transform
from torchvision import datasets, transforms
from torch.utils.data.sampler import RandomSampler
from torchvision.transforms import InterpolationMode  # Import InterpolationMode
from typing import Tuple, Any  # Import typing


class RandomMaskingGenerator:
    def __init__(self, input_size, mask_ratio):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2
        self.height, self.width = input_size
        self.num_patches = self.height * self.width
        self.num_mask = int(mask_ratio * self.num_patches)

    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.num_patches, self.num_mask
        )
        return repr_str

    def __call__(self):
        mask = np.hstack([
            np.zeros(self.num_patches - self.num_mask),
            np.ones(self.num_mask)])
        np.random.shuffle(mask)
        return mask


class Gen_ma(object):
    def __init__(self, is_train, args ):
        self.transform = build_img_transform(is_train, args)
        self.masked_position_generator = RandomMaskingGenerator(
            args.window_size, args.mask_ratio)

    def __call__(self, image):
        return self.transform(image), self.masked_position_generator()

    def __repr__(self):
        repr = "Gen_ma(\n"
        repr += "  transform = %s,\n" % str(self.transform)
        repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
        repr += ")"
        return repr


def build_dataset(is_train: bool, args: Any) -> data.Dataset:
    """Builds the appropriate dataset based on args."""
    transform = Gen_ma(is_train, args)
    if args.data_set.startswith('cifar_S32'):
        dataset = CIFAR_M(args.data_path, train=is_train, transform=transform,
                                        download=True)
    elif args.data_set.startswith('cifar_S224'):
        root = os.path.join(args.data_path, 'CIFAR_S224/Train' if is_train else 'CIFAR_S224/Test')
        dataset = ImageFolder(root, transform=transform)
    elif args.data_set.startswith('imagenet'):
        root = os.path.join(args.data_path, 'Imagenet/Train' if is_train else 'Imagenet/test')
        dataset = ImageFolder(root, transform=transform)
    # --- Add Fish Dataset Support ---
    elif args.data_set == 'fish':  # Use a specific name for your dataset
        root = os.path.join("fish_image")  # Directly use the provided data_path
        dataset = ImageFolder(root, transform=transform)  # Use ImageFolder
    # ---------------------------------
    else:
        raise NotImplementedError(f"Dataset {args.data_set} not implemented")
    return dataset


def build_img_transform(is_train: bool, args: Any) -> transforms.Compose:
    """Builds image transformations."""
    if args.data_set.startswith('cifar_S32'):
        resize_im = args.input_size > 32
    elif args.data_set.startswith('cifar_S224'):
        resize_im = False
    elif args.data_set.startswith('fish'): # Added case
        resize_im = True

    mean = (0., 0., 0.)
    std = (1., 1., 1.)
    # --- Data Augmentation for Training ---
    if is_train:
      transform_list = []
      if resize_im:
          crop_pct = 1.0
          size = int(args.input_size / crop_pct)
          # Use newer interpolation API
          transform_list.append(transforms.Resize(size, interpolation=InterpolationMode.BICUBIC))
          transform_list.append(transforms.CenterCrop(args.input_size))
      transform_list.extend([
          transforms.RandomResizedCrop(args.input_size, scale=(0.7, 1.0)),  # Add some cropping
          transforms.RandomHorizontalFlip(),
          transforms.RandomRotation(degrees=15),
          transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
          transforms.ToTensor(),
          transforms.Normalize(mean, std),
      ])
      return transforms.Compose(transform_list)

    # --- Validation Transform ---
    else:
        t = []
        if resize_im:
            crop_pct = 1.0
            size = int(args.input_size / crop_pct)
             # Use newer interpolation API
            t.append(transforms.Resize(size, interpolation=InterpolationMode.BICUBIC))
            t.append(transforms.CenterCrop(args.input_size))
        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(mean, std))
        return transforms.Compose(t)