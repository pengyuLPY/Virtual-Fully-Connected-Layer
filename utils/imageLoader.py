import torch.utils.data as data

from .data_transformer import *
from PIL import Image
import os
import os.path
import numpy as np

from tqdm import tqdm

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    #from torchvision import get_image_backend
    #if get_image_backend() == 'accimage':
    #    return accimage_loader(path)
    #else:
    #    return pil_loader(path)
    return pil_loader(path)



class ImageLabelFolder(data.Dataset):
    class imageLabel:
        def __init__(self, image_path, label_len):
            self.image = image_path
            self.labels = None
            self.label_len = label_len

    def __init__(self, root, proto, transform=None, target_transform=None, sign_imglist=False,loader=default_loader, key_index=0):
        #protoFile = open(proto)
        protoFile = open(proto,encoding="utf8",errors='ignore')
        content = protoFile.readlines()
        self.imageLabel_list = []
        self.label_len = -1
        self.loader = loader
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.sign_imglist = sign_imglist
        errorFile = open('errorFile.log','w')

        for line in tqdm(content):
            line = line.strip()
            line_list = line.split(' ')
            if line_list.__len__()-1 != self.label_len and self.label_len != -1:
                raise(RuntimeError('(imageLoader)LABEL NUMBER ERROR: self.label_len == line_list.__len__()-1'+':'+
                                   str(self.label_len) +'vs'+ str(line_list.__len__() - 1)))

            img_path = self.root+line_list[0]
            if os.path.isfile(img_path) == False:
                errorFile.write(img_path+' '+str(os.path.isfile(img_path))+'\n')
                continue

            cur_imageLabel = self.imageLabel(img_path,line_list.__len__()-1)
            label_list = []
            for i in range(1,line_list.__len__()):
                label_list.append(float(line_list[i]))

            cur_imageLabel.labels = torch.FloatTensor(label_list)
            self.imageLabel_list.append(cur_imageLabel)

        self.labelgenerator(key_index)
        print ('data size is ',self.imageLabel_list.__len__(), 'vs', content.__len__())

    def labelgenerator(self, key_index=0):
        self.labels = torch.FloatTensor((self.imageLabel_list.__len__()))
        for i, imglabel in enumerate(self.imageLabel_list):
            self.labels[i] = imglabel.labels[key_index]



    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """


        cur_imageLabel = self.imageLabel_list[index]

        img = self.loader(cur_imageLabel.image)
        labels = cur_imageLabel.labels
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            labels = self.target_transform(labels)
        if self.sign_imglist:
            return img, labels, cur_imageLabel.image
        else:
            return img, labels

    def __len__(self):
        return len(self.imageLabel_list)







