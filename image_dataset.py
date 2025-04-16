import os
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import random
import torch.nn.functional as F
import numpy as np
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

class SegmentationDataset(Dataset):
    def __init__(self, root_dir_images, root_dir_masks, image_size=256, crop=None, horiztonal_flip=False, 
                 onehot=True, brightness=False, contrast=False, vertical_flip=False, rotate=False, full_resolution=False):
        self.root_dir_images = root_dir_images
        self.root_dir_masks = root_dir_masks
        self.image_size = image_size
        self.crop = crop
        self.onehot = onehot
        self.horizontal_flip = horiztonal_flip
        self.vertical_flip = vertical_flip
        self.brightness = brightness
        self.contrast = contrast
        self.rotate = rotate
        self.full_resolution=full_resolution
        self.image_names = sorted([file for file in os.listdir(self.root_dir_images) if file.endswith(('.png', '.jpg', '.jpeg'))])
        
    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir_images, self.image_names[idx]).replace('\\','/')
        mask_name = os.path.join(self.root_dir_masks, os.path.splitext(self.image_names[idx])[0] + '_mapped_mask.png').replace('\\','/')

        image = Image.open(img_name).convert('RGB')
        mask = Image.open(mask_name).convert('L')
        image, mask = self.apply_transforms(image, mask)
        return {'image': image, 'mask': mask, 'index': idx}

    def apply_transforms(self, image, mask):
        # Resize
        if not self.full_resolution:
            image = TF.resize(image, (self.image_size, self.image_size), interpolation=transforms.InterpolationMode.BILINEAR)
            mask = TF.resize(mask, (self.image_size, self.image_size), interpolation=transforms.InterpolationMode.NEAREST)

        # CenterCrop or RandomCrop
        if self.crop == "center":
            image = TF.center_crop(image, self.image_size)
            mask = TF.center_crop(mask, self.image_size)
        elif self.crop == "random":
            top = random.randint(0, image.height - self.image_size)
            left = random.randint(0, image.width - self.image_size)
            image = TF.crop(image, top, left, self.image_size, self.image_size)
            mask = TF.crop(mask, top, left, self.image_size, self.image_size)

        # RandomHorizontalFlip
        if self.horizontal_flip and random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # RandomVerticalFlip          
        if self.vertical_flip and random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        # Rotate
        if self.rotate and random.random() > 0.5:
            image, mask = TF.rotate(image, 90), TF.rotate(mask, 90)

        # Brightness Adjustment
        if self.brightness and random.random() > 0.5:
            bright_factor = random.uniform(0.9, 1.1)
            image = TF.adjust_brightness(image, bright_factor)

        # Contrast Adjustment
        if self.contrast and random.random() > 0.5:
            cont_factor = random.uniform(0.9, 1.1)
            image = TF.adjust_contrast(image, cont_factor)

        # ToTensor and Normalize for image
        image = TF.to_tensor(image)
        image = TF.normalize(image, [0.5], [0.5])

        # ToTensor for mask
        mask = TF.to_tensor(mask).squeeze() 
        mask = (mask * 255).long()

        unique_mask_values = torch.sort(torch.tensor([128, 161, 226, 201, 172, 77, 76, 44, 221, 156, 189, 126, 127], dtype=torch.long)).values
        mapping = torch.arange(0, len(unique_mask_values))
        mapped_indices = torch.searchsorted(unique_mask_values, mask)
        mask = mapping[mapped_indices]
        if self.onehot:
            mask = F.one_hot(mask, num_classes=13).permute(2, 0, 1)
        return image, mask


    # def one_hot_encode_label(self, label, unique_labels):
    #     mask = torch.isin(label, unique_labels)
    #     contains_false = not torch.all(mask)
    #     if contains_false:
    #         print(f"Mask contains False: {contains_false}")

    #     mapped_indices = torch.searchsorted(unique_labels, label[mask])

    #     unknown_index = 100
    #     mapped_labels = torch.full_like(label, unknown_index, dtype=torch.long)
    #     mapped_labels[mask] = mapped_indices


    #     num_classes = len(unique_labels)
    #     labels_one_hot = torch.nn.functional.one_hot(mapped_labels, num_classes=num_classes)
    #     labels_one_hot = labels_one_hot.permute(3, 0, 1, 2)
    #     return labels_one_hot.float()

# def show_images(images, masks, num_images=3):
# # need to comment mask = self.one_hot_encode_label(mask, unique_labels).squeeze(0)
#     fig, axs = plt.subplots(num_images, 2, figsize=(10, num_images * 5))
#     for i in range(num_images):
#         img = images[i].permute(1, 2, 0).numpy()
#         img = (img * 0.5 + 0.5) * 255 
#         img = img.astype('uint8') 

#         mask = masks[i].squeeze()
#         mask = mask.numpy()

#         axs[i, 0].imshow(img)
#         axs[i, 0].set_title('Image')
#         axs[i, 0].axis('off')

#         axs[i, 1].imshow(mask, cmap='gray') 
#         axs[i, 1].set_title('Mask')
#         axs[i, 1].axis('off')

#     plt.show()

if __name__ == "__main__":
    dataset = SegmentationDataset(
        root_dir_images=r'/home/neurolinku/Projects/Last_workstation/data/cholec_train_test/train/images',
        root_dir_masks=r'/home/neurolinku/Projects/Last_workstation/data/cholec_train_test/train/groundtruth_no_white_color',
        image_size=256,
        full_resolution=True,
        crop=None,
        horiztonal_flip=True,
        vertical_flip=True,
        rotate=True
    )
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)    
    # unique_labels = set()
    # for data in tqdm(dataloader, desc="Processing batches"):
    #     images, masks, idxs = data['image'], data['mask'], data['index']
    #     for mask in masks:
    #         # Convert mask tensor to numpy and update the set with unique elements
    #         unique_labels.update(np.unique(mask.numpy()))

    # print("Unique labels found in masks:", unique_labels)
    # exit()
    batch = next(iter(dataloader))
    images = batch['image']
    masks = batch['mask']
    print(masks.shape)
    print(images.shape)
    # show_images(images, masks, num_images=4)
    
    import numpy as np
    print(np.unique(masks[0].numpy()))

