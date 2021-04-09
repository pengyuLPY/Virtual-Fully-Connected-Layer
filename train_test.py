import os
import shutil
import time
import numpy as np
import torch
from tqdm import tqdm

def train(train_loader, train_loader_MS, model, criterion, optimizer, lr_obj, epoch, train_iter_per_epoch, display, IterationNum, snapshot, snapshot_prefix, gpu=None):

    model.train()

    print('Training length is {} and {}'.format(len(train_loader), len(train_loader_MS)))
    i = -1
    i_ms = -1
    end = time.time()
    train_loader_iter = train_loader.__iter__()
    train_loader_MS_iter = train_loader_MS.__iter__()
    while True:
        i += 1
        i_ms += 1
        #if train_loader_iter.batches_outstanding == 0:
        if i >= len(train_loader):
            train_loader_iter = train_loader.__iter__()
            i = 0
        #if train_loader_MS_iter.batches_outstanding == 0:
        if i_ms >= len(train_loader_MS_iter):
            break
        data,target = train_loader_iter.__next__()
        data_MS,target_MS = train_loader_MS_iter.__next__()
        target_MS_set = set(target_MS.tolist())

        if gpu is None:
            target = target.cuda()
            data = data.cuda()
            target_MS = target_MS.cuda()
            data_MS = data_MS.cuda()
        else:
            target = target.cuda(gpu, non_blocking=True)
            data = data.cuda(gpu, non_blocking=True)
            target_MS = target_MS.cuda(gpu, non_blocking=True)
            data_MS = data_MS.cuda(gpu, non_blocking=True)

        output,output_MS,target_MS_new = model(data, data_MS, target, target_MS, target_MS_set, testing=False, extract_feature=False)

        loss_CA = criterion(output,target)
        loss_MS = criterion(output_MS,target_MS_new)

        loss = loss_CA + loss_MS * 10


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_CA_show = loss_CA.item()
        loss_MS_show = loss_MS.item()
        loss_show = loss.item()

        run_time = time.time() - end
        end = time.time()
        IterationNum[0] += 1

        if IterationNum[0] % display == 0:
            print('Device_ID:' + str(torch.cuda.current_device()) + ', '
                  'Epoch:' + str(epoch) + ', ' + 'Iteration:' + str(IterationNum[0]) + ', ' +
                  'TrainSpeed = '+ str(run_time) + ', ' +
                  'lr = ' + str(optimizer.param_groups[0]['lr']) +', ' +
                  'Trainloss_CA = ' + str(loss_CA_show) + ',' + 
                  'Trainloss_MS = ' + str(loss_MS_show) + ',' + 
                  'Trainloss = ' + str(loss_show) + ',')

        if IterationNum[0] % snapshot == 0:
                save_checkpoint(
                    {
                        'epoch': epoch+1,
                        'state_dict': model.state_dict(),
                        'best_pre1': -1,
                        'optimizer' : optimizer.state_dict()
                    }, False, snapshot_prefix+'_iter_'+str(IterationNum[0])
                )
        if train_iter_per_epoch!=0 and IterationNum[0] % train_iter_per_epoch == 0:
                break;        

def test(test_loader, test_loader_MS, model, criterion, epoch, test_iter = 0, gpu=None):
    model.eval()


    end = time.time()
    loss_CA_show = 0
    loss_MS_show = 0
    loss_show = 0
    prec1_CA = 0
    prec1_MS = 0
    num = 0.0
    i = -1
    i_ms = -1
    print('Testing length is {} and {}'.format(len(test_loader),len(test_loader_MS)))
    test_loader_iter = test_loader.__iter__()
    test_loader_MS_iter = test_loader_MS.__iter__()
    with torch.no_grad():
        while(True):
            i += 1
            i_ms += 1
            if test_iter != 0 and i == test_iter:
                break
            #if test_loader_MS_iter.batches_outstanding == 0:
            if i_ms >= len(test_loader_MS):
                break
            #if test_loader_iter.batches_outstanding == 0:
            if i >= len(test_loader):
                test_loader_iter = test_loader.__iter__()
                i = 0
              
            data,target = test_loader_iter.__next__()
            data_MS,target_MS = test_loader_MS_iter.__next__()
            target_MS_set = set(target_MS.tolist())


            if gpu is None:
                target = target.cuda()
                data = data.cuda()
                target_MS = target_MS.cuda()
                data_MS = data_MS.cuda()
            else:
                target = target.cuda(gpu, non_blocking=True)
                data = data.cuda(gpu, non_blocking=True)
                target_MS = target_MS.cuda(gpu, non_blocking=True)
                data_MS = data_MS.cuda(gpu, non_blocking=True)

            (output0, output1),(output0_MS, output1_MS),target_MS_new = model(data, data_MS, target, target_MS, target_MS_set, testing=True, extract_feature=False)
            loss_t = criterion(output0, target)
            loss_CA_show += loss_t.item()
            loss_t = criterion(output0_MS, target_MS_new)
            loss_MS_show += loss_t.item()
            loss_show = loss_CA_show + loss_MS_show*10 

            prec1_t = accuracy(output1.data, target)
            prec1_CA += prec1_t[0][0]
            prec1_t = accuracy(output1_MS.data, target_MS_new)
            prec1_MS += prec1_t[0][0]


            num += 1


        run_time = time.time() - end
        print('Device_ID:' + str(torch.cuda.current_device()) + ', '
             'Epoch:' + str(epoch) + ', ' +
             'test_iter:' + str(test_iter) + ', ' +
              'TestSpeed = ' + str(run_time) + ', ' +
             'Testloss_CA = ' + str(loss_CA_show / num) + ', ' +
             'Testloss_MS = ' + str(loss_MS_show / num) + ', ' +
             'Testloss = ' + str(loss_show / num) + ', ' +
             'TestAccuracy_CA = ' + str(prec1_CA / num) + ', ' +
             'TestAccuracy_MS = ' + str(prec1_MS / num) + ', ')
    return 0

def extract_feature_per_img(test_loader, test_loader_flip, model, root):
        model.eval()
        pre = 0

        for i,((data_ori,target, imglist),(data_flip,target,imglist2)) in enumerate(zip(test_loader, test_loader_flip)):
                if i % 100 == 0:
                        print (i,'vs',test_loader.__len__())
                data_ori = data_ori.cuda()
                data_var_ori = torch.autograd.Variable(data_ori, volatile=True)
                output_ori = model.forward_feat(data_var_ori, extract_feature=True)
                data_flip = data_flip.cuda()
                data_var_flip = torch.autograd.Variable(data_flip, volatile=True)
                output_flip = model.forward_feat(data_var_flip, extract_feature=True)

                output = torch.cat((output_ori, output_flip), dim=1)
                output_data = output.data.cpu()
                output_np = output_data.numpy()
                output_np = output_np.astype(np.float32)
                for i,img in enumerate(imglist):
                    assert img==imglist2[i]
                    saveName = root+img+'.bin'
                    path,name = os.path.split(saveName)
                    if os.path.exists(path) == False:
                        command = 'mkdir -p ' + path
                        os.system(command)
                    output_np[i,:].tofile(saveName)


def extract_feature(test_loader, model):
        result = None
        model.eval()
        pre = 0
        for i,(data,target) in enumerate(test_loader):
                if i % 100 == 0:
                        print (i,'vs',test_loader.__len__())
                target = target.cuda()
                data = data.cuda()
                output = model.forward_feat(data, extract_feature=True)

                if result is None:
                        size = np.array(output.data.cpu().size())
                        n = size[0]
                        size[0] = test_loader.dataset.__len__()
                        result = torch.FloatTensor(*size).zero_()

                result[pre:pre+n,:] = output.data.cpu().clone()
                pre = pre+n

        return result
        

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res





def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    filename += '.pytorch'
    print ('saving snapshot in',filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')
    print ('saving complete')
