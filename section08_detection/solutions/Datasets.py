import torch
from torch.utils.data.dataset import Dataset

import os
import pandas as pd
import cv2


class CUB200(Dataset):
    def __init__(self, data_set_root, image_size, transform, test_train=1, return_masks=False):

        # Dataset found here
        # https://www.kaggle.com/datasets/wenewone/cub2002011
        
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

    def get_bbox_list(self, data, img_size):
        bbox_array = [data["x"],
                      data["y"],
                      data["width"],
                      data["height"]]

        if (bbox_array[0] + bbox_array[2]) > img_size[1]:
            bbox_array[2] = img_size[1] - bbox_array[0]

        if (bbox_array[1] + bbox_array[3]) > img_size[0]:
            bbox_array[3] = img_size[0] - bbox_array[1]

        return [bbox_array]

    def get_output_tensors(self, data_out):
        if len(data_out["bboxes"]) > 0:
            bbox = torch.FloatTensor(data_out["bboxes"][0]) / self.image_size
            label = data_out["class_labels"][0]
        else:
            bbox = torch.zeros(4)
            label = -1

        return bbox, [label]

    def __getitem__(self, index):
        data_series = self.data_df.iloc("index")[index]
        file_path = data_series["file_path"]
        label = data_series["class"]

        img_path = os.path.join(self.image_root_dir, file_path)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bbox_array = self.get_bbox_list(data_series, image.shape)

        if self.return_masks:
            mask_path = os.path.join(self.mask_root_dir, file_path).split(".jpg")[0] + ".png"
            mask = cv2.imread(mask_path)

            data_out = self.transform(image=image, bboxes=bbox_array, mask=mask, class_labels=[label])
            bbox, label = self.get_output_tensors(data_out)
            mask = (data_out["mask"][:, :, 0] > 100).long()

            return data_out["image"], mask, bbox, label
        else:
            data_out = self.transform(image=image, bboxes=bbox_array, class_labels=[label])
            bbox, label = self.get_output_tensors(data_out)

            return data_out["image"], bbox, label

    def __len__(self):
        return len(self.data_df)