import os
import time
import argparse
import random
import logging

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models
import torch.backends.cudnn as cudnn

from utils import get_logger, AverageMeter, make_datasets, get_lr

seed = 1234

parser = argparse.ArgumentParser(description='PyTorch Benchmarking')
parser.add_argument('--prefix', type=str, default='Prefix')
parser.add_argument('--model_name', type=str, default='resnet18')
parser.add_argument('--pretrained', action='store_true', default=False)
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--batch_size', default=100, type=int)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--step_size', default=30, type=int)
parser.add_argument('--gamma', default=0.1, type=float)
args = parser.parse_args()

root = 'benchmark-pytorch'
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def build_model(model_name, pretrained):
    model = models.__dict__[model_name](num_classes=10, pretrained=pretrained)
    return model


class Trainer(object):
    
    def __init__(self, model, loader, optimizer, criterion, scheduler,
                 device, **kwargs):
        model = model.to(device)
        if device == "cuda":
            model = torch.nn.DataParallel(model)
            cudnn.benchmark = True
        self.model = model
        
        self.loader = loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device
        
        self.output_dir = '{}/output/{}-{}-{}'.format(
            root, args.prefix, self.model.__class__.__name__, os.getpid())   
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        self.logger = get_logger(os.path.join(self.output_dir, 'train.log'))

        self.logger.info('Build Trainer...')
        self.logger.info('Available gpu device count: {}'.format(torch.cuda.device_count()))
        self.logger.info(self.output_dir)
        self.logger.info('Batch size: {}'.format(loader['train'].batch_size))
        self.logger.info('Initial learning rate: {}'.format(optimizer.defaults['lr']))
        self.logger.info('Momentum: {}'.format(optimizer.defaults['momentum']))
        self.logger.info('Step size: {}'.format(scheduler.step_size))
        self.logger.info('Gamma: {}'.format(scheduler.gamma))
    
    def train(self, epochs):
        best_acc = 0.0
        train_acc, train_loss = [], []
        test_acc, test_loss = [], []
        duration = []

        start = time.time()
        self.logger.info('Start training!')
        for epoch in range(1, epochs + 1):
            train_log = self.train_single_step()
            test_log = self.test()
            
            self.scheduler.step(epoch)
            
            msg = '[TRAIN {}] Epoch: {}/{}'.format(
                    self.model.__class__.__name__, epoch, epochs)
            for k, v in train_log.items():
                msg += ' - {}: {:.3f}'.format(k, v)    
            self.logger.info(msg)
            train_acc.append(train_log['accuracy'])
            train_loss.append(train_log['loss'])
            duration.append(train_log['duration'])
            
            msg = '[TEST {}] Epoch: {}/{}'.format(
                    self.model.__class__.__name__, epoch, epochs)
            for k, v in test_log.items():
                msg += ' - {}: {:.3f}'.format(k, v)    
            self.logger.info(msg)
            test_acc.append(test_log['accuracy'])
            test_loss.append(test_log['loss'])
            
            if test_log['accuracy'] > best_acc:
                self.logger.debug('Saving...')
                state = {
                    'model': model.state_dict(),
                    'accuracy': test_log['accuracy'],
                    'epoch': epoch,
                }
                torch.save(state, os.path.join(self.output_dir, 'ckpt.pth'))
                best_acc = test_log['accuracy']
        
        self.save_logs(train_acc, train_loss, test_acc, test_loss)
        
        self.logger.info('Finish training!')
        self.logger.info('Best test accuracy: {}'.format(best_acc))
        self.logger.info('Elapsed time per epoch: {} sec.'.format(np.mean(duration)))
        self.logger.info('Total training time: {} min.'.format((time.time() - start) / 60.))
                
    def save_logs(self, train_acc, train_loss, test_acc, test_loss):
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        x = np.arange(len(train_acc))
        ax1.plot(x, train_acc, label='Train set')
        ax1.plot(x, test_acc, label='Test set')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Accuracy curve.')
        ax1.legend(loc='best')
        
        ax2 = fig.add_subplot(212)
        ax2.plot(x, train_loss, label='Train set')
        ax2.plot(x, test_loss, label='Test set')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title('Loss curve.')
        ax2.legend(loc='best')
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'training_curve.png')
        plt.savefig(save_path, transparent=True)
        
        np.save(os.path.join(self.output_dir, 'train_acc.npy'), train_acc)
        np.save(os.path.join(self.output_dir, 'train_loss.npy'), train_loss)
        np.save(os.path.join(self.output_dir, 'test_acc.npy'), test_acc)
        np.save(os.path.join(self.output_dir, 'test_loss.npy'), test_loss)

    def train_single_step(self):
        running_loss = 0.0
        correct = 0
        total = 0
        duration = []
        
        start = time.time()
        acc_meter = AverageMeter()
        loss_meter = AverageMeter()
        for i, (x, t) in enumerate(self.loader['train']):
            x, t = x.to(self.device), t.to(self.device)
            
            self.optimizer.zero_grad()
            y = self.model(x)
            loss = self.criterion(y, t)
            loss.backward()
            self.optimizer.step()
            
            _, pred = torch.max(y.data, 1)
            total += t.size(0)
            correct += (pred == t).sum().item()
            acc = 100 * correct / total
            
            acc_meter.update(acc, t.size(0))
            loss_meter.update(loss.item(), t.size(0))

        return {'loss': loss_meter.avg,
                'accuracy': acc_meter.avg,
                'duration': time.time() - start,
                'lr': get_lr(self.optimizer)}

    def test(self):
        correct = 0
        total = 0
        
        acc_meter = AverageMeter()
        loss_meter = AverageMeter()
        with torch.no_grad():
            for i, (x, t) in enumerate(self.loader['test']):
                x, t = x.to(self.device), t.to(self.device)
    
                y = self.model(x)
                loss = self.criterion(y, t)
                _, pred = torch.max(y.data, 1)
                total += t.size(0)
                correct += (pred == t).sum().item()
                acc = 100 * correct / total
            
                acc_meter.update(acc, t.size(0))
                loss_meter.update(loss.item(), t.size(0))
                
        return {'loss': loss_meter.avg,
                'accuracy': acc_meter.avg}


if __name__ == '__main__':
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    loader = make_datasets(batch_size=args.batch_size,
			   root=os.path.join(root, 'data'))
    model = build_model(args.model_name, args.pretrained)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    scheduler = optim.lr_scheduler.StepLR(optimizer,
                                          step_size=args.step_size,
                                          gamma=args.gamma)
    
    trainer = Trainer(model, loader, optimizer, criterion, scheduler, device)
    trainer.train(args.epochs)
