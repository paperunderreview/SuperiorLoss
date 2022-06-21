#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2019 Apple Inc. All Rights Reserved.
#
import time
import argparse

import numpy as np
import torch
from torch import Tensor
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.nn.parallel
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tensorboard_logger import log_value

import utils
from dataset.cifar_dataset import CIFAR10WithIdx, CIFAR100WithIdx
from models.wide_resnet import WideResNet28_10, WideResNet16_10
from models.resnet import ResNet18, ResNet34

import math


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=120, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--restart', default=False, const=True, action='store_const',
                    help='Erase log and saved checkpoints and restart training')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate for model parameters', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)',
                    dest='weight_decay')
parser.add_argument('--rand_fraction', default=0.0, type=float, help='Fraction of data we will corrupt')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency')
parser.add_argument('--seed', default=1, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--learn_class_parameters', default=False, const=True, action='store_const',
                    help='Learn temperature per class')
parser.add_argument('--learn_inst_parameters', default=False, const=True, action='store_const',
                    help='Learn temperature per instance')
parser.add_argument('--skip_clamp_data_param', default=False, const=True, action='store_const',
                    help='Do not clamp data parameters during optimization')
parser.add_argument('--lr_class_param', default=0.1, type=float, help='Learning rate for class parameters')
parser.add_argument('--lr_inst_param', default=0.1, type=float, help='Learning rate for instance parameters')
parser.add_argument('--wd_class_param', default=0.0, type=float, help='Weight decay for class parameters')
parser.add_argument('--wd_inst_param', default=0.0, type=float, help='Weight decay for instance parameters')
parser.add_argument('--init_class_param', default=1.0, type=float, help='Initial value for class parameters')
parser.add_argument('--init_inst_param', default=1.0, type=float, help='Initial value for instance parameters')


def adjust_learning_rate(model_initial_lr, optimizer, gamma, step):
    """Sets the learning rate to the initial learning rate decayed by 10 every few epochs.

    Args:
        model_initial_lr (int) : initial learning rate for model parameters
        optimizer (class derived under torch.optim): torch optimizer.
        gamma (float): fraction by which we are going to decay the learning rate of model parameters
        step (int) : number of steps in staircase learning rate decay schedule
    """
    lr = model_initial_lr * (gamma ** step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# my code
class MyCrossEntropyLoss(nn.Module):
    # train loss
    def __init__(self, lam=1, tau = 1.5) -> None:
        super(MyCrossEntropyLoss, self).__init__()
        self.tau = tau
        self.lam = lam

    # def forward(self, input, target: Tensor) -> Tensor:
    #my code
    def forward(self, input, target: Tensor, my_sigma = None) -> Tensor:
        loss = F.cross_entropy(input, target)


        # my code
        sigma  = None
        if my_sigma is not None:
            sigma = torch.tensor(my_sigma).to("cuda")
            beta = (loss - self.tau)/self.lam
            cap = torch.tensor(-(2/math.e)).to("cuda")
            sigma = torch.exp(-self.lambertw(0.5 * torch.max(cap, beta)))
        else: sigma = self.sigma(loss)

        return ((loss - self.tau)*sigma) + (self.lam*torch.pow(torch.log(sigma), 2))

    def sigma(self, loss: Tensor) -> Tensor:
        beta = (loss - self.tau)/self.lam
        cap = torch.tensor(-(2/math.e)).to("cuda")
        return torch.exp(-self.lambertw(0.5 * torch.max(cap, beta)))

    def _taylor_approx(self, z: Tensor) -> Tensor:
        """Compute an approximation of the lambertw function at z.
        Based on the polynomial expansion in https://arxiv.org/pdf/1003.1628.pdf. An empirical comparison of this polynomial
        expansion against the winitzki approximation found that this one is better when z < -0.2649.
        Args:
            z: The input to the lambertw function.
        Returns:
            An estimated value of lambertw(z).
        """
        p2 = 2 * (1. + math.e * z)
        p = torch.sqrt(p2)
        return -1. + p - p2 / 3. + 0.1527777777 * p2 * p

    def _lambertw_winitzki_approx(self, z: Tensor) -> Tensor:
        """Compute an approximation of the lambertw function at z.
        Args:
            z: The input to the lambertw function.
        Returns:
            An estimated value of lambertw(z).
        """
        log1pz = torch.log1p(z)
        return log1pz * (1. - torch.log1p(log1pz) / (2. + log1pz))

    def lambertw(self, z: Tensor) -> Tensor:
        """Approximate the LambertW function value using Halley iteration.
        Args:
            z: The inputs to the LambertW function.
        Returns:
            An approximation of W(z).
        """
        # Make some starting guesses in order to converge faster.
        z0 = torch.where(z < -0.2649, self._taylor_approx(z), self._lambertw_winitzki_approx(z))
        tolerance = 1e-6
        # Perform at most 20 halley iteration refinements of the value (usually finishes in 2)
        for _ in range(20):
            f = z0 - z * torch.exp(-z0)
            z01 = z0 + 1.0000001  # Numerical stability when z0 == -1
            delta = f / (z01 - (z0 + 2.) * f / (2. * z01))
            z0 = z0 - delta
            converged = torch.abs(delta) <= tolerance * torch.abs(z0)
            if torch.all(converged):
                break
        return z0


def get_train_and_val_loader(args):
    """"Constructs data loaders for train and val on CIFAR100

    Args:
        args (argparse.Namespace):

    Returns:
        train_loader (torch.utils.data.DataLoader): data loader for CIFAR100 train data.
        val_loader (torch.utils.data.DataLoader): data loader for CIFAR100 val data.
    """
    print("w16 10 ce 10")
    print('==> Preparing data for CIFAR10..')

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = CIFAR10WithIdx(root='.',
                               train=True,
                               download=True,
                               transform=transform_train,
                               rand_fraction=args.rand_fraction)
    valset = CIFAR10WithIdx(root='.',
                             train=False,
                             download=True,
                             transform=transform_val)
    train_loader = torch.utils.data.DataLoader(trainset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.workers)
    val_loader = torch.utils.data.DataLoader(valset,
                                             batch_size=100,
                                             shuffle=False,
                                             num_workers=args.workers)

    return train_loader, val_loader


def get_model_and_loss_criterion(args):
    """Initializes DNN model and loss function.

    Args:
        args (argparse.Namespace):

    Returns:
        model (torch.nn.Module): DNN model.
        criterion (torch.nn.modules.loss): cross entropy loss
    """
    # print('Building WideResNet28_10')
    # args.arch = 'WideResNet28_10'
    # model = WideResNet28_10(num_classes=args.nr_classes)
    print('Building WideResNet16_10')
    args.arch = 'WideResNet16_10'
    model = WideResNet16_10(num_classes=args.nr_classes)
    # My Code
    # print('Building ResNet34')
    # args.arch = 'ResNet34'
    # model = ResNet34(num_classes=args.nr_classes)
    if args.device == 'cuda':
        model = model.cuda()
        # my code
        # criterion = nn.CrossEntropyLoss().cuda()
        criterion = MyCrossEntropyLoss().cuda()
    else:
        # my code
        # criterion = nn.CrossEntropyLoss()
        criterion = MyCrossEntropyLoss()

    return model, criterion


def validate(args, val_loader, model, criterion, epoch):
    """Evaluates model on validation set and logs score on tensorboard.

    Args:
        args (argparse.Namespace):
        val_loader (torch.utils.data.dataloader): dataloader for validation set.
        model (torch.nn.Module): DNN model.
        criterion (torch.nn.modules.loss): cross entropy loss
        epoch (int): current epoch
    """
    losses = utils.AverageMeter('Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (inputs, target, _) in enumerate(val_loader):
            if args.device == 'cuda':
                inputs = inputs.cuda()
                target = target.cuda()

            # compute output
            logits = model(inputs)
            loss = criterion(logits, target)

            # measure accuracy and record loss
            acc1 = utils.compute_topk_accuracy(logits, target, topk=(1, ))
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1[0].item(), inputs.size(0))

        print('Test-Epoch-{}: Acc:{}, Loss:{}'.format(epoch, top1.avg, losses.avg))

    # Logging results on tensorboard
    log_value('val/accuracy', top1.avg, step=epoch)
    log_value('val/loss', losses.avg, step=epoch)


def train_for_one_epoch(args,
                        train_loader,
                        model,
                        criterion,
                        optimizer,
                        epoch,
                        global_iter,
                        optimizer_data_parameters,
                        data_parameters,
                        config):
    """Train model for one epoch on the train set.

    Args:
        args (argparse.Namespace):
        train_loader (torch.utils.data.dataloader): dataloader for train set.
        model (torch.nn.Module): DNN model.
        criterion (torch.nn.modules.loss): cross entropy loss.
        optimizer (torch.optim.SGD): optimizer for model parameters.
        epoch (int): current epoch.
        global_iter (int): current iteration count.
        optimizer_data_parameters (tuple SparseSGD): SparseSGD optimizer for class and instance data parameters.
        data_parameters (tuple of torch.Tensor): class and instance level data parameters.
        config (dict): config file for the experiment.

    Returns:
        global iter (int): updated iteration count after 1 epoch.
    """

    # Initialize counters
    losses = utils.AverageMeter('Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')

    # Unpack data parameters
    optimizer_class_param, optimizer_inst_param = optimizer_data_parameters
    class_parameters, inst_parameters = data_parameters

    # Switch to train mode
    model.train()
    start_epoch_time = time.time()
    for i, (inputs, target, index_dataset) in enumerate(train_loader):
        global_iter = global_iter + 1
        inputs, target = inputs.to(args.device), target.to(args.device)

        # Flush the gradient buffer for model and data-parameters
        optimizer.zero_grad()
        if args.learn_class_parameters:
            optimizer_class_param.zero_grad()
        if args.learn_inst_parameters:
            optimizer_inst_param.zero_grad()

        # my code
        # Compute logits
        # f = open("std.txt", "a")
        # model.mean_std = []
        logits = model(inputs)
        # logits = model(inputs, f, epoch)
        # f.close()


        if args.learn_class_parameters or args.learn_inst_parameters:
            # Compute data parameters for instances in the minibatch
            class_parameter_minibatch = class_parameters[target]
            inst_parameter_minibatch = inst_parameters[index_dataset]
            data_parameter_minibatch = utils.get_data_param_for_minibatch(
                                            args,
                                            class_param_minibatch=class_parameter_minibatch,
                                            inst_param_minibatch=inst_parameter_minibatch)

            # Compute logits scaled by data parameters
            logits = logits / data_parameter_minibatch

        # my code
        # loss = criterion(logits, target)
        loss = criterion(logits, target, my_sigma=model.mean_std)

        # Apply weight decay on data parameters
        if args.learn_class_parameters or args.learn_inst_parameters:
            loss = utils.apply_weight_decay_data_parameters(args, loss,
                                                            class_parameter_minibatch=class_parameter_minibatch,
                                                            inst_parameter_minibatch=inst_parameter_minibatch)

        # Compute gradient and do SGD step
        loss.backward()
        optimizer.step()
        if args.learn_class_parameters:
            optimizer_class_param.step()
        if args.learn_inst_parameters:
            optimizer_inst_param.step()

        # Clamp class and instance level parameters within certain bounds
        if args.learn_class_parameters or args.learn_inst_parameters:
            utils.clamp_data_parameters(args, class_parameters, config, inst_parameters)

        # Measure accuracy and record loss
        acc1 = utils.compute_topk_accuracy(logits, target, topk=(1, ))
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1[0].item(), inputs.size(0))

        # Log stats for data parameters and loss every few iterations
        # i have commented out the following code
        # if i % args.print_freq == 0:
        #     utils.log_intermediate_iteration_stats(args, class_parameters, epoch,
        #                                            global_iter, inst_parameters,
        #                                            losses, top1)

    # Print and log stats for the epoch
    # print('Time for epoch: {}'.format(time.time() - start_epoch_time))
    print('Train-Epoch-{}: Acc:{}, Loss:{}'.format(epoch, top1.avg, losses.avg))
    log_value('train/accuracy', top1.avg, step=epoch)
    log_value('train/loss', losses.avg, step=epoch)

    # print(f"Loss at epoch {epoch}: {losses.avg}")

    # my code
    return global_iter, losses.avg


def main_worker(args, config):
    """Trains model on ImageNet using data parameters

    Args:
        args (argparse.Namespace):
        config (dict): config file for the experiment.
    """
    global_iter = 0
    learning_rate_schedule = np.array([80, 100, 160])

    # Create model
    model, loss_criterion = get_model_and_loss_criterion(args)

    # Define optimizer
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # Get train and validation dataset loader
    train_loader, val_loader = get_train_and_val_loader(args)

    # Initialize class and instance based temperature
    (class_parameters, inst_parameters,
     optimizer_class_param, optimizer_inst_param) = utils.get_class_inst_data_params_n_optimizer(
                                                        args=args,
                                                        nr_classes=args.nr_classes,
                                                        nr_instances=len(train_loader.dataset),
                                                        device='cuda'
                                                        )
    # my code
    losses = []
    # Training loop
    # my code
    for epoch in range(args.start_epoch, args.epochs):
    # for epoch in range(1):

        # Adjust learning rate for model parameters
        if epoch in learning_rate_schedule:
            adjust_learning_rate(model_initial_lr=args.lr,
                                 optimizer=optimizer,
                                 gamma=0.1,
                                 step=np.sum(epoch >= learning_rate_schedule))

        # Train for one epoch
        # my code
        global_iter, loss = train_for_one_epoch(
                            args=args,
                            train_loader=train_loader,
                            model=model,
                            criterion=loss_criterion,
                            optimizer=optimizer,
                            epoch=epoch,
                            global_iter=global_iter,
                            optimizer_data_parameters=(optimizer_class_param, optimizer_inst_param),
                            data_parameters=(class_parameters, inst_parameters),
                            config=config)
        losses.append(loss)

        # Evaluate on validation set
        validate(args, val_loader, model, loss_criterion, epoch)

        # Save artifacts
        utils.save_artifacts(args, epoch, model, class_parameters, inst_parameters)

        # i have commented out the following code
        # Log temperature stats over epochs
        # if args.learn_class_parameters:
        #     utils.log_stats(data=torch.exp(class_parameters),
        #                     name='epochs_stats_class_parameter',
        #                     step=epoch)
        # if args.learn_inst_parameters:
        #     utils.log_stats(data=torch.exp(inst_parameters),
        #                     name='epoch_stats_inst_parameter',
        #                     step=epoch)

        if args.rand_fraction > 0.0:
            # We have corrupted labels in the train data; plot instance parameter stats for clean and corrupt data
            nr_corrupt_instances = int(np.floor(len(train_loader.dataset) * args.rand_fraction))
            # Corrupt data is in the top-fraction of dataset
            utils.log_stats(data=torch.exp(inst_parameters[:nr_corrupt_instances]),
                            name='epoch_stats_corrupt_inst_parameter',
                            step=epoch)
            utils.log_stats(data=torch.exp(inst_parameters[nr_corrupt_instances:]),
                            name='epoch_stats_clean_inst_parameter',
                            step=epoch)
    # my code
    print("Average of losses: ", np.mean(losses))


def main():
    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.log_dir = './logs_CL_CIFAR'
    args.save_dir = './weights_CL_CIFAR'
    args.nr_classes = 10 # Number classes in CIFAR100
    utils.generate_log_dir(args)
    utils.generate_save_dir(args)

    config = {}
    config['clamp_inst_sigma'] = {}
    config['clamp_inst_sigma']['min'] = np.log(1/20)
    config['clamp_inst_sigma']['max'] = np.log(20)
    config['clamp_cls_sigma'] = {}
    config['clamp_cls_sigma']['min'] = np.log(1/20)
    config['clamp_cls_sigma']['max'] = np.log(20)
    utils.save_config(args.save_dir, config)

    # Set seed for reproducibility
    utils.set_seed(args)

    # Simply call main_worker function
    main_worker(args, config)


if __name__ == '__main__':
    main()
