#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import math
import torch
import errno
import random
import numpy as np
import matplotlib.pyplot as plt


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True





def update_learning_rate(scheduler, optimizer):
    scheduler.step()
    lr = optimizer.param_groups[0]['lr']

class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        # assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)

class Logger(object):
	'''Save training process to log file with simple plot function.'''
	def __init__(self, fpath, title=None, resume=False): 
		self.file = None
		self.resume = resume
		self.title = '' if title == None else title
		if fpath is not None:
			if resume: 
				self.file = open(fpath, 'r') 
				name = self.file.readline()
				self.names = name.rstrip().split('\t')
				self.numbers = {}
				for _, name in enumerate(self.names):
					self.numbers[name] = []

				for numbers in self.file:
					numbers = numbers.rstrip().split('\t')
					for i in range(0, len(numbers)):
						self.numbers[self.names[i]].append(numbers[i])
				self.file.close()
				self.file = open(fpath, 'a')  
			else:
				self.file = open(fpath, 'w')

	def set_names(self, names):
		if self.resume: 
			pass
		# initialize numbers as empty list
		self.numbers = {}
		self.names = names
		for _, name in enumerate(self.names):
			self.file.write(name)
			self.file.write('\t')
			self.numbers[name] = []
		self.file.write('\n')
		self.file.flush()


	def append(self, numbers):
		assert len(self.names) == len(numbers), 'Numbers do not match names'
		for index, num in enumerate(numbers):
			self.file.write("{0:.6f}".format(num))
			self.file.write('\t')
			self.numbers[self.names[index]].append(num)
		self.file.write('\n')
		self.file.flush()

	def plot(self, names=None):   
		names = self.names if names == None else names
		numbers = self.numbers
		for _, name in enumerate(names):
			x = np.arange(len(numbers[name]))
			plt.plot(x, np.asarray(numbers[name]))
		plt.legend([self.title + '(' + name + ')' for name in names])
		plt.grid(True)

	def close(self):
		if self.file is not None:
			self.file.close()


def mkNew_dir(path):
	'''make dir if not exist'''
	try:
		os.makedirs(path)
	except OSError as exc:  # Python >2.5
		if exc.errno == errno.EEXIST and os.path.isdir(path):
			pass
		else:
			raise  

def Loss_plt(train_perf_s, losses_s, val_s, data_dir, metrics=2): # 2 for SSIM
    epochs = [item[0] for item in train_perf_s]
    val = [item[metrics] for item in val_s]
    train = [item[metrics] for item in train_perf_s]
        
    losse_epch = [item[0] for item in losses_s]
    totalAdv_loss = [item[0] for item in losses_s]
    totalGAn_loss = [item[1] for item in losses_s]
        
        
    fig1, ax1 = plt.subplots(dpi=100)
    ax1.plot(epochs, train, 'b', label='Training SSIM')
    ax1.plot(epochs, val, 'r', label='validation SSIM')
    ax1.set_title('Training and Validation SSIM')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('SSIM')
    ax1.legend()
    fig1.savefig(data_dir + '/Training and Validation'+str(metrics)+'.png', dpi=fig1.dpi)
        
    fig2, ax2 = plt.subplots(dpi=100)
    ax2.plot(losse_epch, totalAdv_loss, 'b', label='Discriminator loss')
    ax2.plot(losse_epch, totalGAn_loss, 'g', label='Generator loss')
    ax2.set_title('Generator and Discriminator loss')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.legend()
    fig2.savefig(data_dir + '/Generator and Discriminator'+str(metrics)+'.png', dpi=fig2.dpi)