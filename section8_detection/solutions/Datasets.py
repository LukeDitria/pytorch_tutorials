import torch
import torchvision
import torchvision.transforms.functional as ft

from torch.utils.data.dataset import Dataset

import os
import numpy as np
from PIL import Image, ImageOps
import pandas as pd
import copy


class CUB200(Dataset):
    def __init__(self, data_set_root, transform, test_train=1, image_size=128, return_masks=False):
        
        class_list_path = os.path.join(data_set_root, "CUB_200_2011/CUB_200_2011/image_class_labels.txt")
        self.data_df = pd.read_csv(class_list_path, sep=" ", names=["index", "class"])

        data_list_path = os.path.join(data_set_root, "CUB_200_2011/CUB_200_2011/images.txt")
        cub200_df = pd.read_csv(data_list_path, sep=" ", names=["index", "file_path"])

        bbox_list_path = os.path.join(data_set_root, "CUB_200_2011/CUB_200_2011/bounding_boxes.txt")
        bbox_df = pd.read_csv(bbox_list_path, sep=" ", names=["index", "x", "y", "width", "height"])

        # Use custom test/train split 1/0
        split_df = pd.read_csv("test_train_split.txt", sep=" ", names=["index", "split"])

        self.data_df = self.data_df.merge(cub200_df, left_on='index', right_on='index')
        self.data_df = self.data_df.merge(bbox_df, left_on='index', right_on='index')
        self.data_df = self.data_df.merge(split_df, left_on='index', right_on='index')
        
        self.data_df = self.data_df[self.data_df.split != test_train]

        self.return_masks = return_masks
        self.image_size = image_size
        self.transform = transform
        self.data_set_root = data_set_root
        self.image_root_dir = os.path.join(self.data_set_root, "CUB_200_2011/CUB_200_2011/images")
        self.mask_root_dir = os.path.join(self.data_set_root, "segmentations")

    def __getitem__(self, index):
        data_series = self.data_df.iloc("index")[index]
        file_path = data_series["file_path"]
        label = data_series["class"]

        img_path = os.path.join(self.image_root_dir, file_path)
        img = Image.open(img_path).convert('RGB')
        target_size = max(img.size)

        # Create Bounding Box array
        # min_x, min_y, height, width
        # Compensate for the padding by adding offsets
        padding_offset = np.array([(target_size - img.size[0])/2,
                                   (target_size - img.size[1])/2,
                                   0, 0])

        bbox_array = np.array([data_series["x"],
                               data_series["y"], 
                               data_series["width"], 
                               data_series["height"]])
        
        bbox_array = torch.FloatTensor(bbox_array + padding_offset)
        bbox_array = bbox_array/target_size

        # Pad the image to be a square
        img = ImageOps.pad(img, (target_size, target_size))
        # Remove this if you are using v2 transforms
        img = self.transform(img)

        # Pytorch version >= 0.15 have v2 transforms that work on both the images and the bounding boxes
        # as well as the segmentation masks
        # Use the following transform instead of the above if you wish to try it out!

        # Convert the bounding box array to a Pytorch bounding box object!
        # (You'll need to do the same with the segmentation mask with datapoints.Mask)
        # https://pytorch.org/blog/extending-torchvisions-transforms-to-object-detection-segmentation-and-video-tasks/

        # bbox_array = torchvision.datapoints.BoundingBox(bbox_array,
        #                                                 format="XYWH",
        #                                                 spatial_size=(target_size, target_size))
        #
        # img, bbox_array = self.transform(img, bbox_array)
        # bbox_array /= self.image_size

        if self.return_masks:
            mask_path = os.path.join(self.mask_root_dir, file_path)
            mask = (ft.to_tensor(Image.open(mask_path)) > 0.5).long()[0]

            return img, mask, bbox_array, torch.tensor(label)
        else:
            return img, bbox_array, torch.tensor(label)

    def __len__(self):
        return len(self.data_df)