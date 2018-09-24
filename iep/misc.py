#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 rvedantam3 <vrama91@vt.edu>
#
# Distributed under terms of the MIT license.

"""
Some misc functions.
"""
import json
import numpy as np
import torch

import iep.utils as utils
from iep.models import ModuleNet
from iep.models import Seq2Seq
from iep.models import LstmModel
from iep.models import CnnLstmModel
from iep.models import CnnLstmSaModel
from iep.models.seq import SeqModel
import iep.preprocess

from torch.autograd import Variable


def parse_int_list(s):
  return tuple(int(n) for n in s.split(','))


def apply_supervision_on_loss(loss_tensor, supervision):
  loss_tensor = loss_tensor.squeeze().float()
  supervision = supervision.float()

  if loss_tensor.dim() == 0:
    raise ValueError("Provided loss tensor cannot be a scalar")

  if loss_tensor.size(0) != supervision.data.size(0):
    raise ValueError(
        "Supervision and loss tensor must have the same batch dimensions")

  return torch.sum(torch.mul(loss_tensor, supervision)) / (torch.sum(supervision) + 1e-8)


def get_state(m):
  if m is None:
    return None
  state = {}
  for k, v in m.state_dict().items():
    state[k] = v.clone()
  return state


def get_program_prior(args):
  vocab = load_vocab(args.vocab_json)
  if args.program_prior_start_from is not None:
    prior, kwargs = load_program_prior(
      args.program_prior_start_from
    )
  else:
    kwargs = {
        'decoder_vocab_size': len(vocab['program_token_to_idx']),
        'wordvec_dim': args.rnn_wordvec_dim,
        'hidden_dim': args.rnn_hidden_dim,
        'rnn_num_layers': args.rnn_num_layers,
        'rnn_dropout': args.rnn_dropout,
    }
    prior = SeqModel(**kwargs)
  prior.cuda()
  prior.train()
  return prior, kwargs


def get_question_reconstructor(args):
  vocab = load_vocab(args.vocab_json)
  if args.program_generator_start_from is not None:
    qr, kwargs = load_question_reconstructor(
        args.question_reconstructor_start_from)
  else:
    kwargs = {
        'encoder_vocab_size': len(vocab['program_token_to_idx']),
        'decoder_vocab_size': len(vocab['question_token_to_idx']),
        'wordvec_dim': args.rnn_wordvec_dim,
        'hidden_dim': args.rnn_hidden_dim,
        'rnn_num_layers': args.rnn_num_layers,
        'rnn_dropout': args.rnn_dropout,
        'max_reinforce_sample_length': args.max_question_length,
    }
    qr = Seq2Seq(**kwargs)
  qr.cuda()
  qr.train()
  return qr, kwargs


def get_program_generator(args):
  vocab = load_vocab(args.vocab_json)
  if args.program_generator_start_from is not None:
    pg, kwargs = load_program_generator(args.program_generator_start_from)
    cur_vocab_size = pg.encoder_embed.weight.size(0)
    if cur_vocab_size != len(vocab['question_token_to_idx']):
      print('Expanding vocabulary of program generator')
      pg.expand_encoder_vocab(vocab['question_token_to_idx'])
      kwargs['encoder_vocab_size'] = len(vocab['question_token_to_idx'])
  else:
    kwargs = {
        'encoder_vocab_size': len(vocab['question_token_to_idx']),
        'decoder_vocab_size': len(vocab['program_token_to_idx']),
        'wordvec_dim': args.rnn_wordvec_dim,
        'hidden_dim': args.rnn_hidden_dim,
        'rnn_num_layers': args.rnn_num_layers,
        'rnn_dropout': args.rnn_dropout,
        'max_reinforce_sample_length': args.max_program_length,
    }
    pg = Seq2Seq(**kwargs)
  pg.cuda()
  pg.train()
  return pg, kwargs


def get_execution_engine(args):
  vocab = load_vocab(args.vocab_json)
  if args.execution_engine_start_from is not None:
    ee, kwargs = load_execution_engine(args.execution_engine_start_from)
    # TODO: Adjust vocab?
  else:
    kwargs = {
        'vocab': vocab,
        'feature_dim': parse_int_list(args.feature_dim),
        'stem_batchnorm': args.module_stem_batchnorm == 1,
        'stem_num_layers': args.module_stem_num_layers,
        'module_dim': args.module_dim,
        'module_residual': args.module_residual == 1,
        'module_batchnorm': args.module_batchnorm == 1,
        'classifier_proj_dim': args.classifier_proj_dim,
        'classifier_downsample': args.classifier_downsample,
        'classifier_fc_layers': parse_int_list(args.classifier_fc_dims),
        'classifier_batchnorm': args.classifier_batchnorm == 1,
        'classifier_dropout': args.classifier_dropout,
    }
    ee = ModuleNet(**kwargs)
  ee.cuda()
  ee.train()
  return ee, kwargs


def get_baseline_model(args):
  vocab = load_vocab(args.vocab_json)
  if args.baseline_start_from is not None:
    model, kwargs = load_baseline(args.baseline_start_from)
  elif args.model_type == 'LSTM':
    kwargs = {
        'vocab': vocab,
        'rnn_wordvec_dim': args.rnn_wordvec_dim,
        'rnn_dim': args.rnn_hidden_dim,
        'rnn_num_layers': args.rnn_num_layers,
        'rnn_dropout': args.rnn_dropout,
        'fc_dims': parse_int_list(args.classifier_fc_dims),
        'fc_use_batchnorm': args.classifier_batchnorm == 1,
        'fc_dropout': args.classifier_dropout,
    }
    model = LstmModel(**kwargs)
  elif args.model_type == 'CNN+LSTM':
    kwargs = {
        'vocab': vocab,
        'rnn_wordvec_dim': args.rnn_wordvec_dim,
        'rnn_dim': args.rnn_hidden_dim,
        'rnn_num_layers': args.rnn_num_layers,
        'rnn_dropout': args.rnn_dropout,
        'cnn_feat_dim': parse_int_list(args.feature_dim),
        'cnn_num_res_blocks': args.cnn_num_res_blocks,
        'cnn_res_block_dim': args.cnn_res_block_dim,
        'cnn_proj_dim': args.cnn_proj_dim,
        'cnn_pooling': args.cnn_pooling,
        'fc_dims': parse_int_list(args.classifier_fc_dims),
        'fc_use_batchnorm': args.classifier_batchnorm == 1,
        'fc_dropout': args.classifier_dropout,
    }
    model = CnnLstmModel(**kwargs)
  elif args.model_type == 'CNN+LSTM+SA':
    kwargs = {
        'vocab': vocab,
        'rnn_wordvec_dim': args.rnn_wordvec_dim,
        'rnn_dim': args.rnn_hidden_dim,
        'rnn_num_layers': args.rnn_num_layers,
        'rnn_dropout': args.rnn_dropout,
        'cnn_feat_dim': parse_int_list(args.feature_dim),
        'stacked_attn_dim': args.stacked_attn_dim,
        'num_stacked_attn': args.num_stacked_attn,
        'fc_dims': parse_int_list(args.classifier_fc_dims),
        'fc_use_batchnorm': args.classifier_batchnorm == 1,
        'fc_dropout': args.classifier_dropout,
    }
    model = CnnLstmSaModel(**kwargs)
  if model.rnn.token_to_idx != vocab['question_token_to_idx']:
    # Make sure new vocab is superset of old
    for k, v in model.rnn.token_to_idx.items():
      assert k in vocab['question_token_to_idx']
      assert vocab['question_token_to_idx'][k] == v
    for token, idx in vocab['question_token_to_idx'].items():
      model.rnn.token_to_idx[token] = idx
    kwargs['vocab'] = vocab
    model.rnn.expand_vocab(vocab['question_token_to_idx'])
  model.cuda()
  model.train()
  return model, kwargs


def set_mode(mode, models):
  assert mode in ['train', 'eval']
  for m in models:
    if m is None:
      continue
    if mode == 'train':
      m.train()
    if mode == 'eval':
      m.eval()


def check_accuracy(args, program_prior, question_reconstructor,
                   program_generator, execution_engine, baseline_model, loader):
  set_mode('eval', [
      program_prior, question_reconstructor, program_generator,
      execution_engine, baseline_model
  ])
  num_correct, num_samples = 0, 0
  for idx_batch, batch in enumerate(loader):
    questions, _, feats, answers, programs, _, _ = batch

    questions_var = Variable(questions.cuda(), volatile=True)
    feats_var = Variable(feats.cuda(), volatile=True)
    if programs[0] is not None:
      programs_var = Variable(programs.cuda(), volatile=True)

    scores = None  # Use this for everything but PG and Prior
    neg_logprob = None
    if args.model_type == 'Prior':
      num_correct=None  # Dont use this for the prior.
      neg_logprob = []
      for i in range(programs.size(0)):
        program_neg_logprob = program_prior(
            Variable(programs[i:i + 1].cuda(), volatile=True))
        neg_logprob.append(float(program_neg_logprob.data.cpu().numpy()))
    elif args.model_type == 'PG':
      vocab = load_vocab(args.vocab_json)
      programs_pred = program_generator.reinforce_sample(questions_var)
      programs_pred = programs_pred.detach().data.cpu().numpy()
      for program_idx in range(programs_pred.shape[0]):
        this_programs = programs_pred[program_idx]
        this_programs = list(this_programs[this_programs!=0])
        program_pred_str = iep.preprocess.decode(this_programs,
                                                 vocab['program_idx_to_token'])
        program_str = iep.preprocess.decode(programs[program_idx],
                                            vocab['program_idx_to_token'])
        assert program_pred_str[0] == '<START>', "First program in the predicted sample must be the start token."
        if program_pred_str == program_str:
          num_correct += 1
        num_samples += 1
    elif args.model_type == 'EE':
      scores = execution_engine(feats_var, programs_var)
    elif args.model_type == 'PG+EE':
      programs_pred = program_generator.reinforce_sample(
          questions_var, argmax=True)
      scores = execution_engine(feats_var, programs_pred)
    elif args.model_type in ['LSTM', 'CNN+LSTM', 'CNN+LSTM+SA']:
      scores = baseline_model(questions_var, feats_var)

    if scores is not None:
      _, preds = scores.data.cpu().max(1)
      num_correct += (preds == answers).sum()
      num_samples += preds.size(0)

    if num_samples >= args.num_val_samples:
      break

  set_mode('train', [
      program_prior, question_reconstructor, program_generator,
      execution_engine, baseline_model
  ])
  if num_correct is not None:
    acc = float(num_correct) / num_samples
  elif neg_logprob is not None:
    acc = -1 * np.mean(neg_logprob)
  else:
    raise RuntimeError
  return acc

def load_vocab(path):
  with open(path, 'r') as f:
    vocab = json.load(f)
    vocab['question_idx_to_token'] = utils.invert_dict(vocab['question_token_to_idx'])
    vocab['program_idx_to_token'] = utils.invert_dict(vocab['program_token_to_idx'])
    vocab['answer_idx_to_token'] = utils.invert_dict(vocab['answer_token_to_idx'])
  # Sanity check: make sure <NULL>, <START>, and <END> are consistent
  assert vocab['question_token_to_idx']['<NULL>'] == 0
  assert vocab['question_token_to_idx']['<START>'] == 1
  assert vocab['question_token_to_idx']['<END>'] == 2
  assert vocab['program_token_to_idx']['<NULL>'] == 0
  assert vocab['program_token_to_idx']['<START>'] == 1
  assert vocab['program_token_to_idx']['<END>'] == 2
  return vocab


def load_cpu(path):
  """
  Loads a torch checkpoint, remapping all Tensors to CPU
  """
  print('Loading checkpoint from file: %s' % (path))
  return torch.load(path, map_location=lambda storage, loc: storage)


def load_program_prior(path):
  checkpoint = load_cpu(path)
  kwargs = checkpoint['program_prior_kwargs']
  state = checkpoint['program_prior_state']
  model = SeqModel(**kwargs)
  model.load_state_dict(state)
  return model, kwargs


def load_question_reconstructor(path):
  checkpoint = load_cpu(path)
  kwargs = checkpoint['question_reconstructor_kwargs']
  state = checkpoint['question_reconstructor_state']
  model = Seq2Seq(**kwargs)
  model.load_state_dict(state)
  return model, kwargs

def load_program_generator(path):
  checkpoint = load_cpu(path)
  kwargs = checkpoint['program_generator_kwargs']
  state = checkpoint['program_generator_state']
  model = Seq2Seq(**kwargs)
  model.load_state_dict(state)
  return model, kwargs


def load_execution_engine(path, verbose=True):
  checkpoint = load_cpu(path)
  kwargs = checkpoint['execution_engine_kwargs']
  state = checkpoint['execution_engine_state']
  kwargs['verbose'] = verbose
  model = ModuleNet(**kwargs)
  model.load_state_dict(state)
  return model, kwargs

def load_baseline(path):
  model_cls_dict = {
    'LSTM': LstmModel,
    'CNN+LSTM': CnnLstmModel,
    'CNN+LSTM+SA': CnnLstmSaModel,
  }
  checkpoint = load_cpu(path)
  baseline_type = checkpoint['baseline_type']
  kwargs = checkpoint['baseline_kwargs']
  state = checkpoint['baseline_state']

  model = model_cls_dict[baseline_type](**kwargs)
  model.load_state_dict(state)
  return model, kwargs
