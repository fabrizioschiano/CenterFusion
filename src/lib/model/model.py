from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torchvision.models as models
import torch
import torch.nn as nn
import os

from .networks.dla import DLASeg
from .networks.resdcn import PoseResDCN
from .networks.resnet import PoseResNet
from .networks.dlav0 import DLASegv0
from .networks.generic_network import GenericNetwork

str_print = "[log - model.py] "

_network_factory = {
  'resdcn': PoseResDCN,
  'dla': DLASeg,
  'res': PoseResNet,
  'dlav0': DLASegv0,
  'generic': GenericNetwork
}

def create_model(arch, head, head_conv, opt=None):
  print(str_print + 'in create_model()')
  # the following line tries to find the number of layers from the "name" of the architecture used. E.g. dla_34 will have 34 layers
  print(str_print + 'arch (including the num_layers): ' + arch)
  num_layers = int(arch[arch.find('_') + 1:]) if '_' in arch else 0
  print(str_print + 'num_layers: ' + str(num_layers))
  # the following line removes the _num_layers from the arch string. E.g. 'dla_34' will become 'dla'
  arch = arch[:arch.find('_')] if '_' in arch else arch
  print(str_print + 'arch (stripped of the num_layers): ' + arch)
  model_class = _network_factory[arch]
  # e.g., model_class: <class 'model.networks.dla.DLASeg'>
  print(str_print + 'model_class: ' + str(model_class))
  model = model_class(num_layers, heads=head, head_convs=head_conv, opt=opt)
  return model

def load_model(model, model_path, opt, optimizer=None):
  print(str_print + 'in load_model()')
  start_epoch = 0
  # Load the model into the CPU
  checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
  print(str_print+"type(checkpoint): ", type(checkpoint))
  print(str_print+"checkpoint.keys(): ", checkpoint.keys())
  # An alternative to previous command would be to load the model into the GPU 1 with
  # checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage.cuda(0))
  print(str_print + 'loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
  state_dict_ = checkpoint['state_dict']
  print(str_print+"type(state_dict_): ", type(state_dict_))
  state_dict = {}
   
  # convert data_parallal to model
  for k in state_dict_:
    if k.startswith('module') and not k.startswith('module_list'):
      state_dict[k[7:]] = state_dict_[k]
    else:
      state_dict[k] = state_dict_[k]
  model_state_dict = model.state_dict()

  # check loaded parameters and created model parameters
  for k in state_dict:
    if k in model_state_dict:
      # print(str_print + k + " is in model_state_dict")
      # print("   " + str_print + "state_dict[k].shape: " + str(state_dict[k].shape))
      # print("   " + str_print + "model_state_dict[k].shape: " + str(model_state_dict[k].shape))

      if (state_dict[k].shape != model_state_dict[k].shape) or \
        (opt.reset_hm and k.startswith('hm') and (state_dict[k].shape[0] in [80, 1])):
        print(str_print + "checking for loaded parameters")
        if opt.reuse_hm:
          print('Reusing parameter {}, required shape{}, '\
                'loaded shape{}.'.format(
            k, model_state_dict[k].shape, state_dict[k].shape))
          # todo: bug in next line: both sides of < are the same
          if state_dict[k].shape[0] < state_dict[k].shape[0]:
            model_state_dict[k][:state_dict[k].shape[0]] = state_dict[k]
          else:
            model_state_dict[k] = state_dict[k][:model_state_dict[k].shape[0]]
          state_dict[k] = model_state_dict[k]
        
        elif opt.warm_start_weights:
          print(str_print + "warming_start_weights")
          try:
            print('Partially loading parameter {}, required shape{}, '\
                  'loaded shape{}.'.format(
              k, model_state_dict[k].shape, state_dict[k].shape))
            if state_dict[k].shape[1] < model_state_dict[k].shape[1]:
              model_state_dict[k][:,:state_dict[k].shape[1]] = state_dict[k]
            else:
              model_state_dict[k] = state_dict[k][:,:model_state_dict[k].shape[1]]
            state_dict[k] = model_state_dict[k]
          except:
            print('Skip loading parameter {}, required shape{}, '\
                'loaded shape{}.'.format(
                k, model_state_dict[k].shape, state_dict[k].shape))
            state_dict[k] = model_state_dict[k]
        
        else:
          print(str_print + 'Skip loading parameter {}, required shape{}, '\
                'loaded shape{}.'.format(
            k, model_state_dict[k].shape, state_dict[k].shape))
          state_dict[k] = model_state_dict[k]
    else:
      print('Drop parameter {}.'.format(k))
  for k in model_state_dict:
    if not (k in state_dict):
      print('No param {}.'.format(k))
      state_dict[k] = model_state_dict[k]
  # The following lines loads the parameters for each layer of the model 'model'
  # The parameters of a model in pytorch are the "learnable" parameters (i.e. weights and biases) of an torch.nn.Module model. 
  # They can be accessed with model.parameters(). 
  # A state_dict is simply a Python dictionary object that maps each layer to its parameter tensor. 
  # Note that only layers with learnable parameters (convolutional layers, linear layers, etc.) and registered buffers 
  # (batchnorm’s running_mean) have entries in the model’s state_dict. Optimizer objects (torch.optim) also have a state_dict, 
  # which contains information about the optimizer’s state, as well as the hyperparameters used.
  model.load_state_dict(state_dict, strict=False)

  # freeze backbone network
  if opt.freeze_backbone:
    print(str_print + "backbone network is frozen")
    for (name, module) in model.named_children():
      if name in opt.layers_to_freeze:
        for (name, layer) in module.named_children():
          for param in layer.parameters():
            param.requires_grad = False

  # resume optimizer parameters
  if optimizer is not None and opt.resume:
    print(str_print + "trying to resume optimizer parameters")
    if 'optimizer' in checkpoint:
      start_epoch = checkpoint['epoch']
      start_lr = opt.lr
      for step in opt.lr_step:
        if start_epoch >= step:
          start_lr *= 0.1
      for param_group in optimizer.param_groups:
        param_group['lr'] = start_lr
      print(str_print + 'Resumed optimizer with start lr', start_lr)
    else:
      print(str_print + 'No optimizer parameters in checkpoint.')
  if optimizer is not None:
    print(str_print + 'optimizer is not None')
    return model, optimizer, start_epoch
  else:
    print(str_print + 'optimizer is None')
    return model

def save_model(path, epoch, model, optimizer=None):
  if isinstance(model, torch.nn.DataParallel):
    state_dict = model.module.state_dict()
  else:
    state_dict = model.state_dict()
  data = {'epoch': epoch,
          'state_dict': state_dict}
  if not (optimizer is None):
    data['optimizer'] = optimizer.state_dict()
  torch.save(data, path)

