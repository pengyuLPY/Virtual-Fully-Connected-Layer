import sys
from torch.utils.data.distributed import DistributedSampler
import torch.backends.cudnn as cudnn
import torch.distributed as dist


from train_test import *
from utils.imageLoader import ImageLabelFolder
import utils.data_transformer as transforms
import utils.target_transformer as target_transformer
import utils.util as Logger 
from utils.lr_policy import *
from utils.batch_sampler import BalancedBatchSampler


from config import *


if __name__ == '__main__':
    print ('Connecting...')
    if distributed:
        dist.init_process_group(
                                backend=dist_backend, 
                                init_method=dist_url,
                                world_size=worlds_size,
                                 rank=rank
                                )
    print ('Connecting complete')
    sys.stdout = Logger.Logger(snapshot_prefix+'_'+str(time.time())+'.log')
    print ('Data loading...')
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
        target_transform=target_transformer.ToInt()
       )
    train_dataset = ImageLabelFolder(
        root=trainRoot, proto=trainProto,
        transform=transforms.Compose([
            transforms.CenterCropWithOffset(150,150,0,20,3,3),
            transforms.RandomHorizontalFlip(),
            transforms.Scale((224,224)),
            transforms.ToTensor(),
            normalize,
        ]),
        target_transform=target_transformer.ToInt()
       )

    test_dataset_MS = ImageLabelFolder(
        root=testRoot_MS, proto=testProto_MS,
        transform=transforms.Compose([
            transforms.CenterCropWithOffset(150,150,0,20,0,0),
            transforms.Scale((224,224)),
            transforms.ToTensor(),
            normalize,
        ]),
        target_transform=target_transformer.ToInt()
       )

        
    train_dataset_MS = ImageLabelFolder(
        root=trainRoot_MS, proto=trainProto_MS,
        transform=transforms.Compose([
            transforms.CenterCropWithOffset(150,150,0,20,3,3),
            transforms.RandomHorizontalFlip(),
            transforms.Scale((224,224)),
            transforms.ToTensor(),
            normalize,
        ]),
        target_transform=target_transformer.ToInt()
       )


    if distributed:
        train_sampler = DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batchSize, shuffle=(train_shuffle and train_sampler is None),
        num_workers = workers, pin_memory=True, sampler = train_sampler, drop_last=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_batchSize, shuffle=test_shuffle,
        num_workers = workers, pin_memory=True, drop_last=True
    )



    train_batch_sampler = BalancedBatchSampler(dataset=train_dataset_MS, n_samples=n_samples, n_min_samples=n_min_samples, batch_size=train_batchSize_MS, shuffle=True)
    train_loader_MS = torch.utils.data.DataLoader(
        train_dataset_MS,  num_workers = workers, pin_memory=True, batch_sampler=train_batch_sampler
    )
    test_batch_sampler = BalancedBatchSampler(dataset=test_dataset_MS, n_samples=n_samples, n_min_samples=n_min_samples, batch_size=test_batchSize_MS, shuffle=True)
    test_loader_MS = torch.utils.data.DataLoader(
        test_dataset_MS,  num_workers = workers, pin_memory=True, batch_sampler=test_batch_sampler
    )


    print ('Data loading complete')
#################################  MODEL INIT ##################
    print ('Model Initing...')
    best_prec1 = 0
    cudnn.benchmark = cudnn_use

    if distributed:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        model.feature_extraction_net = torch.nn.DataParallel(model.feature_extraction_net).cuda()
        model.fc = torch.nn.DataParallel(model.fc).cuda()
        model.fc2 = model.fc2.cuda()

    criterion = torch.nn.CrossEntropyLoss().cuda()
    lr_obj = lr_class()
    optimizer = torch.optim.SGD(model.parameters(), lr_obj.base_lr,
                                momentum=momentum,
                                weight_decay=weight_decay)
    if resume:
        if os.path.isfile(resume_file):
            print("checkpoint => loading checkpoint '{}'".format(resume_file))
            checkpoint = torch.load(resume_file)
            start_epoch = checkpoint['epoch']
            IterationNum[0] = checkpoint['iteration'] 
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("checkpoint => loaded checkpoint '{}' (epoch {})"
                  .format(resume, checkpoint['epoch']))
        else:
            print("checkpoint => no checkpoint found at '{}'".format(resume))
            exit()
    if pretrained:
            if os.path.isfile(pretrained_file):
                    print("pretrain => loading checkpoint '{}'".format(pretrained_file))
                    checkpoint = torch.load(pretrained_file)
                    parameter = checkpoint['state_dict']
                    model.load_state_dict(parameter, False)
            else:
                    print("pretrain => no checkpoint found at '{}'".format(pretrained_file))
                    exit()

    print ('Model Initing complete')

#################################  TRAINING  ##################
    if test_init:
        print ('Testing...')
        prec1 = test(test_loader, test_loader_MS, model, criterion, epoch=0, test_iter=test_iter)


    epoch = -1
    lr_obj = lr_class()
    for epoch in range(start_epoch, max_epoch+1):
#    while IterationNum[0] < max_iter:
            #epoch += 1
        print ('Training... epoch:',epoch)
        adjust_learning_rate(lr_obj, optimizer, epoch)
        if distributed:
            train_sampler.set_epoch(epoch)

        train(train_loader, train_loader_MS, model, criterion, optimizer,lr_obj, epoch, train_iter_per_epoch, display, IterationNum, snapshot, snapshot_prefix)
        print ('Testing... epoch:',epoch)
        prec1 = test(test_loader, test_loader_MS, model, criterion, epoch=epoch, test_iter=test_iter)

        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint(
            {
                'epoch': epoch+1,
                'iteration': IterationNum[0],
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer' : optimizer.state_dict()
            }, is_best, snapshot_prefix+'_epoch_'+str(epoch)
        )
        

            
    print ('Training complete')

