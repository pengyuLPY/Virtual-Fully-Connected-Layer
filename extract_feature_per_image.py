import sys
import random
import os
from base_code.distributed import *
from train_test import *
from base_code.imageLoader import ImageLabelFolder
import base_code.data_transformer as transforms
import base_code.target_transformer as target_transformer
import base_code.util as Logger 
import layers.basic_layer
import numpy as np
import torch.backends.cudnn as cudnn

def extract_feature_main(testRoot, testProto, test_batchSize, pretrained_file, model, root):
#################################  DATA LOAD  ##################
    print (testProto, 'Data loading...')
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])
    test_dataset = ImageLabelFolder(
        root=testRoot, proto=testProto,
        transform=transforms.Compose([
            transforms.CenterCropWithOffset(150,150,0,20,0,0),
            transforms.Scale((224,224)),
            transforms.ToTensor(),
            normalize,
        ]),
        target_transform=target_transformer.ToInt(),
        sign_imglist=True
       )

        
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_batchSize, shuffle=False,
        num_workers = 1, pin_memory=True
    )

    test_dataset_flip = ImageLabelFolder(
        root=testRoot, proto=testProto,
        transform=transforms.Compose([
            transforms.CenterCropWithOffset(150,150,0,20,0,0),
            transforms.RandomHorizontalFlip(flip_prob=2),
            transforms.Scale((224,224)),
            transforms.ToTensor(),
            normalize,
        ]),
        target_transform=target_transformer.ToInt(),
        sign_imglist=True
       )

        
    test_loader_flip = torch.utils.data.DataLoader(
        test_dataset_flip, batch_size=test_batchSize, shuffle=False,
        num_workers = 1, pin_memory=True
    )

    print ('Data loading complete')

    #################################  TRAINING  ##################

    print ('Feature Extracting...')
    extract_feature_per_img(test_loader, test_loader_flip, model, root)

def SaveToBin(feature, save, imglist=None):
        feature_np = feature.numpy(np.float32)
        if imglist is None:
            path,name = os.path.split(save)
            if os.path.exists(path) == False:
                command = 'mkdir -p ' + path
                os.system(command)        
            feature_np.tofile(save)
        else:
            if feature_np.shape[0] != imglist.__len__():
                print ('feature_np.shape[0] != imglist.__len__()',feature_np.shape[0], imglist.__len__())
                exit();
            for i,img in enumerate(imglist):
                img = img.split(' ')[0]
                img = save+img+'.bin'
                path,name = os.path.split(img)
                if os.path.exists(path) == False:
                    command = 'mkdir -p ' + path
                    os.system(command)        
                feature_np[i,:].tofile(img)

        print (feature_np.shape )

if __name__ == '__main__':
        cudnn.benchmark = True

        test_BatchSize = 25
        #################################  MODEL INIT ##################
        print ('Model Initing...')
        from config import model
        pretrained_file = r'./snapshot/resnet101_epoch_16.pytorch'
        sign_imglist=True
        model.feature_extraction_net = torch.nn.DataParallel(model.feature_extraction_net).cuda()
        model.fc = torch.nn.DataParallel(model.fc).cuda()
        model.fc2 = model.fc2.cuda()
        if os.path.isfile(pretrained_file):
                print("pretrain => loading checkpoint '{}'".format(pretrained_file))
                checkpoint = torch.load(pretrained_file)
                parameters = checkpoint['state_dict']
                model.load_state_dict(parameters)
                print("pretrain => loaded checkpoint '{}' (epoch {})"
                        .format(True, checkpoint['epoch']))
        else:
                print("pretrain => no checkpoint found at '{}'".format(pretrained_file))
                exit()

        print ('Model Initing complete')

        testRoot_list = [
                '/home/pengyu.lpy/dataset/'
                ]
        testProto_list = [
                '/home/pengyu.lpy/dataset/cfp-dataset/Align_180_220_imagelist.txt'
                ]
        save_list = [
                './feature/CFP_resnet50net_arcFace_epoch_16_512/'
                ]

        if testRoot_list.__len__() != testProto_list.__len__():
                print ('testRoot_list.__len__() != testProto_list.__len__()',testRoot_list.__len__() ,'vs', testProto_list.__len__())
                exit();
        if testRoot_list.__len__() != save_list.__len__():
                print ('testRoot_list.__len__() != save_list.__len__()',testRoot_list.__len__() ,'vs', save_list.__len__())
                exit();        

        for ind in range(testRoot_list.__len__()):
                testRoot = testRoot_list[ind]
                testProto = testProto_list[ind]
                save = save_list[ind]
                feature = extract_feature_main(testRoot, testProto, test_BatchSize, pretrained_file, model, save)
                print ('------------------------------' )

print ('complete')



