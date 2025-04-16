import os
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import random
import torch.nn.functional as F

class MaskOnlyDataset(Dataset):
    def __init__(self, root_dir_masks, image_size=256, crop=None, onehot=True):
        self.root_dir_masks = root_dir_masks
        self.image_size = image_size
        self.crop = crop
        self.onehot = onehot
        self.mask_names = sorted([file for file in os.listdir(self.root_dir_masks) if file.endswith(('.png', '.jpg', '.jpeg'))])

    def __len__(self):
        return len(self.mask_names)

    def __getitem__(self, idx):
        mask_name = os.path.join(self.root_dir_masks, self.mask_names[idx]).replace('\\','/')

        mask = Image.open(mask_name).convert('L')
        mask = self.apply_transforms(mask)

        return {'mask': mask, 'index': idx}

    def apply_transforms(self, mask):
        # Resize
        mask = TF.resize(mask, (self.image_size, self.image_size), interpolation=transforms.InterpolationMode.NEAREST)

        # CenterCrop or RandomCrop
        if self.crop == "center":
            mask = TF.center_crop(mask, self.image_size)
        elif self.crop == "random":
            top = random.randint(0, mask.height - self.image_size)
            left = random.randint(0, mask.width - self.image_size)
            mask = TF.crop(mask, top, left, self.image_size, self.image_size)

        # ToTensor for mask
        mask = TF.to_tensor(mask).squeeze()
        mask = (mask * 255).long()

        # Adjust this according to the unique values present in your mask files
        unique_mask_values = torch.sort(torch.tensor([128, 161, 226, 201, 172, 77, 76, 44, 221, 156, 189, 126, 127], dtype=torch.long)).values
        mapping = torch.arange(0, len(unique_mask_values))
        mapped_indices = torch.searchsorted(unique_mask_values, mask)
        mask = mapping[mapped_indices]

        if self.onehot:
            mask = F.one_hot(mask, num_classes=len(unique_mask_values)).permute(2, 0, 1)

        return mask
