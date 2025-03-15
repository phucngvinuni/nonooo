import os
import os.path
import random

from torchvision.datasets.vision import VisionDataset
from PIL import Image
from typing import Any, Callable, cast, Dict, List, Optional, Tuple


def has_file_allowed_extension(filename: str, extensions: Tuple[str, ...]) -> bool:
    """Checks if a file is an allowed extension."""
    return filename.lower().endswith(extensions)


def is_image_file(filename: str) -> bool:
    """Checks if a file is an allowed image extension."""
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def make_dataset(
    directory: str,
    class_to_idx: Dict[str, int],
    extensions: Optional[Tuple[str, ...]] = None,
    is_valid_file: Optional[Callable[[str], bool]] = None,
) -> List[Tuple[str, int]]:
    """Generates a list of (sample, class_index) tuples from a directory."""
    instances = []
    directory = os.path.expanduser(directory)

    # Use a default is_valid_file function if none provided.
    if extensions is not None:
        def is_valid_file_func(x: str) -> bool:
            return has_file_allowed_extension(x, extensions)
    elif is_valid_file is not None:
        is_valid_file_func = is_valid_file
    else:
      raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")

    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file_func(path):  # Use the determined function
                    item = (path, class_index)
                    instances.append(item)
    return instances


class DatasetFolder(VisionDataset):
    """A generic data loader."""

    def __init__(
            self,
            root: str,
            loader: Callable[[str], Any],
            extensions: Optional[Tuple[str, ...]] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        classes, class_to_idx = self._find_classes(self.root)
        samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file)
        if len(samples) == 0:
            msg = f"Found 0 files in subfolders of: {self.root}\n"
            if extensions is not None:
                msg += f"Supported extensions are: {','.join(extensions)}"
            raise RuntimeError(msg)

        self.loader = loader
        self.extensions = extensions
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    def _find_classes(self, dir: str) -> Tuple[List[str], Dict[str, int]]:
        """Finds the class folders in a dataset."""
        # --- Modification for Fish Dataset ---
        if "" in os.listdir(dir)[0].lower(): # added .lower()
           classes = [d.name for d in os.scandir(dir) if d.is_dir() and d.name.startswith("")]
           class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        # ---------------------------------------
        else: #Keep the original logic
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()  # Ensure consistent order!
        # --- Modification for Fish Dataset ---
        if "" in os.listdir(dir)[0].lower():
            class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        # ----------------------------------------
        else: # Keep the original logic
            # class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)} # OLD
            classes = ['plane', 'car', 'bird', 'cat', 'deer', # Keep the original logic for other datasets.
               'dog', 'frog', 'horse', 'ship', 'truck']   #
            class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}#
        return classes, class_to_idx

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        try: # Added try except block
            sample = self.loader(path)
        except (IOError, OSError) as e:
            print(f"Error loading image {path}: {e}.  Skipping.")
            return self.__getitem__( (index + 1) % len(self))  # Return next sample

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target


    def __len__(self) -> int:
        return len(self.samples)


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path: str) -> Any:
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path: str) -> Any:
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class ImageFolder(DatasetFolder):
    """A generic data loader where the images are arranged as described above."""

    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super().__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file)
        self.imgs = self.samples