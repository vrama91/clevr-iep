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
import torch
from torch.nn.functional import log_softmax

def sparse_softmax_cross_entropy_with_logits(
    logits=None, labels=None, reduction=None):
  """Compute cross entropy loss, given logits."""
  if logits.dim() - labels.dim() != 1:
    raise ValueError('Labels expected to be sparse.')
  if logits.dim() > 3:
    raise ValueError('Only logits upto 3-D supported')

  if logits.dim() == 3:
    size_0, size_1, _ = logits.size()
    assert labels.size(0) == size_0 and labels.size(1) == size_1
    logits_flat = logits.contiguous().view(logits.size(0)*logits.size(1), -1)
    labels_flat = labels.contiguous().view(-1)
  else:
    logits_flat = logits
    labels_flat = labels

  log_probs = log_softmax(logits_flat, dim=1)

  negative_log_likelihood = -1 * torch.gather(
      log_probs, 1, labels_flat.unsqueeze(1))

  if logits.dim() == 3:
    negative_log_likelihood = negative_log_likelihood.view(size_0, size_1)

  return negative_log_likelihood

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
  vocab = utils.load_vocab(args.vocab_json)
  if args.program_prior_start_from is not None:
    prior, kwargs = utils.load_program_prior(
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
  vocab = utils.load_vocab(args.vocab_json)
  if args.program_generator_start_from is not None:
    qr, kwargs = utils.load_question_reconstructor(
        args.question_reconstructor_load_from)
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
  vocab = utils.load_vocab(args.vocab_json)
  if args.program_generator_start_from is not None:
    pg, kwargs = utils.load_program_generator(args.program_generator_start_from)
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
  vocab = utils.load_vocab(args.vocab_json)
  if args.execution_engine_start_from is not None:
    ee, kwargs = utils.load_execution_engine(args.execution_engine_start_from)
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
  vocab = utils.load_vocab(args.vocab_json)
  if args.baseline_start_from is not None:
    model, kwargs = utils.load_baseline(args.baseline_start_from)
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