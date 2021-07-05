import torch
from imutils import paths
import cv2
from torch.utils.data import Dataset
import os
from src.preprocessing.enhancment import cv2_enhance as enh
from src.utils.utils import extract_bboxes


class XrayData(Dataset):
    def __init__(self, root_paths, transforms=None, enhanced=False, segmented=True, gamma=1):
        self.images = list(paths.list_images(root_paths))
        self.transforms = transforms
        self.enhanced = enhanced
        self.gamma = gamma
        self.segmented = segmented
        self.LABELS = {'Normal': 0, 'Covid': 1, "Lung_Opacity": 2, 'Pneumonia': 3}
        self.COUNT = {'Normal': 0, 'Covid': 0, "Lung_Opacity": 0, 'Pneumonia': 0}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        img = cv2.imread(image_path)
        if self.gamma != 1:
            img = enh.adjust_gamma(img, gamma=self.gamma)

        if self.enhanced:
            img = enh.enhance(img)

        if self.segmented:
            p = image_path.split('/')
            p = p[0] + '/' + p[1] + '/' + p[2] + '/' + p[3] + '/Masked/' + p[5] + '/' + p[6]
            if os.path.exists(p):
                # print("/kaggle/working/"+image_path.split('/')[-4]+"/Masked/"+image_path.split('/')[-2]+'/'+image_path.split('/')[-1])
                mask = cv2.imread(p)
                box = extract_bboxes(mask)
                img = img[max(0, box[0]):min(445, box[2]), max(0, box[1]):min(512, box[3])]

        label = self.get_label(image_path.split('/')[-1])
        y_label = self.get_num_label(label)

        if self.transforms:
            img = self.transforms(img)

        return img, y_label

    @staticmethod
    def get_label(label):
        if label[0] == 'N':
            return 'Normal'
        if label[0] == 'C':
            return 'Covid'
        if label[0] == 'L':
            return 'Lung_Opacity'
        if label[0] == 'P' or label[0] == 'V':
            return 'Pneumonia'

    def get_num_label(self, label):
        return torch.tensor(self.LABELS[label])

    def get_count(self):
        return self.COUNT
