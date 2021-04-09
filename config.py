import CNNs.resnet_identityMapping_face as resnet

worlds_size = 1
distributed = worlds_size > 1
workers = 32
cudnn_use = True
dist_url='tcp://gpu1051.nt12:23456'
dist_backend='nccl'
rank=0


n_samples = 10
n_min_samples = 10
num_classes = 1000 #84282
label_dict = {i:i%num_classes for i in range(84282)}
model = resnet.resnet101(feature_len=512,num_classes_1=10575, num_classes_2=num_classes, label_dict=label_dict)
#snapshot_prefix = r'experiment/verify/Resnet101/casia'
snapshot_prefix = r'./snapshot/resnet101'
snapshot=500000000

test_init=True
display = 100


max_epoch=16
start_epoch = 0
IterationNum = [0]
test_iter = 0
train_iter_per_epoch = 0
resume = False
resume_file = r'./snapshot/resnet101_epoch_9.pytorch'
pretrained = True
pretrained_file = r'resnet101_epoch_16.pytorch' # r'/home/pengyu.lpy/code/pytorch/experiment/pytorchTest_mnist/resnet18_minist_iter_1000'

trainRoot = r'/home/pengyu.lpy/dataset/CASIA/Aligned_180_220/'
trainProto = r'/home/pengyu.lpy/dataset/CASIA/all_CASIA.txt'
train_shuffle = True
train_batchSize = 50 #100

testRoot = r'/home/pengyu.lpy/dataset/CASIA/Aligned_180_220/'
testProto = r'/home/pengyu.lpy/dataset/CASIA/test_CASIA.txt'
test_shuffle = False
test_batchSize = 40


trainRoot_MS = r'/home/pengyu.lpy/dataset/Cleaned_MSCeleb1M_Norm180_220/'
trainProto_MS = r'/home/pengyu.lpy/dataset/Cleaned_MSCeleb1M_Norm180_220/meta_withoutFiltering/recognition_meta_all.txt'
train_batchSize_MS = 700 #100

testRoot_MS = r'/home/pengyu.lpy/dataset/Cleaned_MSCeleb1M_Norm180_220/'
testProto_MS = r'/home/pengyu.lpy/dataset/Cleaned_MSCeleb1M_Norm180_220/meta_withoutFiltering/recognition_meta_test.txt'
test_batchSize_MS = 200



#trainProto = testProto
#trainProto_MS = testProto
#testProto_MS = testProto
#testRoot_MS = testRoot
#trainRoot_MS = testRoot
#test_iter = 10
#train_iter_per_epoch = 100
#test_init = True
#display = 1


class lr_class:
    def __init__(self):
        self.base_lr = 0.01
        self.gamma = 0.1
        self.lr_policy = "multistep"
        self.steps = [12,14,15] #[4800,5400,5800]

momentum = 0.9
weight_decay = 0.0005
