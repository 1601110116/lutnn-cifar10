import os
import time
import torch
from torch import nn
from torch import optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from lookup_model import resnet_cifar
from lookup_model.module_amm import (ModuleAMM, Conv2dAMM, LinearAMM)


os.environ['CUDA_VISIBLE_DEVICES'] = "0"

nolut_ckpt_dir = 'pretrained_models/resnet20-12fca82f.th'
save_dir = 'lutnn_temp_save'
conv_config = {
    'Hin': 32,
    'Win': 32,
    'subD_k3': 9,
    'subD_k1': 4,
    'npts': 16,
    'temperature': 1.0,
}
batch_size = 32
num_epochs = 300
save_every = 10
print_freq = 20


def convert_lut_state(nolut_state, model):
    local_state = {}
    for name, param in nolut_state.items():
        assert isinstance(name, str)
        if name.startswith('module.'):
            name = name[7:]
        local_state.update({name: param})
    lut_state = {}
    for name, m in model.named_modules():
        if isinstance(m, ModuleAMM):
            weight = local_state.pop(name + '.weight')
            lut_state[name + '.base_module.weight'] = weight
            bias = local_state.pop(name + '.bias', None)
            if bias is not None:
                lut_state[name + '.base_module.bias'] = bias
            lut_state[name + '.pts'] = m.pts
            lut_state[name + '.temperature'] = m.temperature
    lut_state.update(local_state)
    return lut_state


def lutnn_init(model: nn.Module):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    cluster_train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize
        ])),
        batch_size=1024, shuffle=True,
        num_workers=4, pin_memory=True)
    x, target = next(iter(cluster_train_loader))
    x = x.cuda()
    for m in model.modules():
        if isinstance(m, ModuleAMM):
            m.need_lutnn_init = True
    model.eval()
    model(x)
    model.train()
    torch.save(model.state_dict(), 'clustered.pth')


best_prec1 = 0


def main():
    global best_prec1

    model = resnet_cifar.resnet20(conv_config=conv_config)
    model.cuda()

    nolut_state = torch.load(nolut_ckpt_dir)['state_dict']
    lut_state = convert_lut_state(nolut_state, model)
    model.load_state_dict(lut_state)
    lutnn_init(model)

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True)
    criterion = nn.CrossEntropyLoss().cuda()

    param_groups = [{'params': [], 'names': [], 'lr': 1e-3},
                    {'params': [], 'names': [], 'lr': 1e-1}]
    for name, param in model.named_parameters():
        if 'temperature' in name:
            param_groups[1]['params'].append(param)
            param_groups[1]['names'].append(name)
        else:
            param_groups[0]['params'].append(param)
            param_groups[0]['names'].append(name)

    # optimizer = optim.Adam(model.parameters(), 1e-3)
    optimizer = optim.Adam(param_groups, lr=1e-3)
    print(optimizer)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=0)

    validate(val_loader, model, criterion)
    for epoch in range(0, num_epochs):

        # train for one epoch
        # print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        print('group0 lr: {:.5e}, group1 lr: {:.5e}'.format(
            optimizer.param_groups[0]['lr'],
            optimizer.param_groups[1]['lr']))
        train(train_loader, model, criterion, optimizer, epoch)
        lr_scheduler.step()

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        if epoch > 0 and epoch % save_every == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, filename=os.path.join(save_dir, 'checkpoint.th'))

        save_checkpoint({
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, filename=os.path.join(save_dir, 'model.th'))


def train(train_loader, model, criterion, optimizer, epoch):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input_var = input.cuda()
        target_var = target

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))


def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input_var = input.cuda()
            target_var = target.cuda()

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time, loss=losses,
                          top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()

