import sys
import os
from base_code.distributed import *
#from base_code.train_test import *
from train_test import *
import torch.backends.cudnn as cudnn 
from base_code.imageLoader import ImageLabelFolder
import base_code.data_transformer as transforms
import base_code.target_transformer as target_transformer
import base_code.util as Logger 
import layers.basic_layer
import numpy as np
import base_code.parallel.data_parallel as data_parallel

def extract_feature_main(testRoot, testProto, test_batchSize, pretrained_file, model, flip_prob=-1):
#################################  DATA LOAD  ##################
    print (testProto, 'Data loading...')
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])
    test_dataset = ImageLabelFolder(
        root=testRoot, proto=testProto,
        transform=transforms.Compose([
            transforms.CenterCropWithOffset(150,150,0,20,0,0, ignore_fault=True),
            transforms.RandomHorizontalFlip(flip_prob=flip_prob),
            transforms.Scale((224,224)),
            transforms.ToTensor(),
            normalize,
        ]),
        target_transform=target_transformer.ToInt()
       )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_batchSize, shuffle=False,
        num_workers = 1, pin_memory=True
    )


    print ('Data loading complete')

    #################################  TRAINING  ##################

    print ('Feature Extracting...')
    feature = extract_feature(test_loader, model)
    print ('Feature Extracting complete', feature.size())
    return feature

def SaveToBin(feature, save):
        feature_np = feature.numpy().astype(np.float32)
        path,name = os.path.split(save)
        if os.path.exists(path) == False:
                command = 'mkdir -p ' + path
                os.system(command)        
        feature_np.tofile(save)
        print (feature_np.shape )

if __name__ == '__main__':
        cudnn.benchmark = True

        test_BatchSize = 240
        #################################  MODEL INIT ##################
        print ('Model Initing...')
        from config import model
        pretrained_file = r'./snapshot/resnet101_epoch_16.pytorch'
        model.feature_extraction_net = torch.nn.DataParallel(model.feature_extraction_net).cuda()
        model.fc = torch.nn.DataParallel(model.fc).cuda()
        model.fc2 = model.fc2.cuda()
        if os.path.isfile(pretrained_file):
                print("pretrain => loading checkpoint '{}'".format(pretrained_file))
                checkpoint = torch.load(pretrained_file)
                parameters = checkpoint['state_dict']
                model.load_state_dict(parameters, strict=True)
                print("pretrain => loaded checkpoint '{}' (epoch {})"
                        .format(True, checkpoint['epoch']))
        else:
                print("pretrain => no checkpoint found at '{}'".format(pretrained_file))
                exit()

        print ('Model Initing complete')

        testRoot_list = [
                '/home/pengyu.lpy/dataset/lfw/lfw_manual_remove_error_detection_align/',
                '/home/pengyu.lpy/dataset/IJB/IJB-A/align_image_180_220_errorDetect200/',
                '/mnt/pengyu.lpy/dataset/IJB/IJB_release_something/IJBB/align_180_220/',
                '/mnt/pengyu.lpy/dataset/IJB/IJB_release_something/IJBC/align_180_220/',
                '/home/pengyu.lpy/dataset/',
                '/home/pengyu.lpy/dataset/'
                   ]
        testProto_list = [
                '/home/pengyu.lpy/dataset/lfw/lfw_manual_remove_error_detection_align/lfw_align_list.txt',
                 '/home/pengyu.lpy/dataset/IJB/IJB-A/labels/idxs/img_list.txt',
                '/mnt/pengyu.lpy/code/tools/face_test/ijbb_ijbc/IJBB/meta/ijbb_name_fakeLabel.txt',
                '/mnt/pengyu.lpy/code/tools/face_test/ijbb_ijbc/IJBC/meta/ijbc_name_fakeLabel.txt',
                '/home/pengyu.lpy/dataset/MegaFace/meta_official/set1/target.txt',
                '/home/pengyu.lpy/dataset/MegaFace/meta_official/set1/disturb.txt'
                   ]
        save_list = [
                './bin/lfw_resnet50net_arcFace_epoch_16.bin',
                './bin/ijba_resnet50net_arcFace_epoch_16.bin',
                './bin/ijbb_resnet50net_arcFace_epoch_16.bin',
                './bin/ijbc_resnet50net_arcFace_epoch_16.bin',
               './bin/megaFaceSet1_resnet50net_arcFace_epoch_16_1024/target.bin',
               './bin/megaFaceSet1_resnet50net_arcFace_epoch_16_1024/disturb.bin',
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
                feature_ori = extract_feature_main(testRoot, testProto, test_BatchSize, pretrained_file, model, flip_prob=-1)
                feature_flip = extract_feature_main(testRoot, testProto, test_BatchSize, pretrained_file, model, flip_prob=2)
                feature = torch.cat((feature_ori, feature_flip), dim=1)
                print (testProto,':', feature.size())
                SaveToBin(feature, save)
                print ('------------------------------') 

print ('complete')



