#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 rvedantam3 <vrama91@vt.edu>
#
# Distributed under terms of the MIT license.

"""

"""
from absl import logging
import torch
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from collections import namedtuple

from iep.misc import sparse_softmax_cross_entropy_with_logits

dims_collection = namedtuple('DimsCollection', ['V_out', 'D', 'H', 'L', 'N', 'T_out'])


class SeqModel(nn.Module):
  def __init__(self,
    decoder_vocab_size=100,
    wordvec_dim=300,
    hidden_dim=256,
    rnn_num_layers=2,
    rnn_dropout=0,
    null_token=0,
    start_token=1,
    end_token=2,
  ):
    super(SeqModel, self).__init__()
    self.decoder_embed = nn.Embedding(decoder_vocab_size, wordvec_dim)
    self.decoder_rnn = nn.LSTM(wordvec_dim, hidden_dim, rnn_num_layers,
                               dropout=rnn_dropout, batch_first=True)
    self.decoder_linear = nn.Linear(hidden_dim, decoder_vocab_size)
    self.NULL = null_token
    self.START = start_token
    self.END = end_token
    self.multinomial_outputs = None

  def get_dims(self, y):
    V_out = self.decoder_embed.num_embeddings
    D = self.decoder_embed.embedding_dim
    H = self.decoder_rnn.hidden_size
    L = self.decoder_rnn.num_layers

    N = y.size(0)
    T_out = y.size(1) if y is not None else None

    return dims_collection(V_out, D, H, L, N, T_out)

  def before_rnn(self, x, replace=0):
    # TODO: Use PackedSequence instead of manually plucking out the last
    # non-NULL entry of each sequence; it is cleaner and more efficient.
    N, T = x.size()
    idx = torch.LongTensor(N).fill_(T - 1)

    # Find the last non-null element in each sequence. Is there a clean
    # way to do this?
    x_cpu = x.cpu()
    for i in range(N):
      for t in range(T - 1):
        if x_cpu.data[i, t] != self.NULL and x_cpu.data[i, t + 1] == self.NULL:
          idx[i] = t
          break
    idx = idx.type_as(x.data)
    x[x.data == self.NULL] = replace
    return x, Variable(idx)

  def decoder(self, y, h0=None, c0=None):
    _dims = self.get_dims(y=y)

    if _dims.T_out > 1:
      y, _ = self.before_rnn(y)
    y_embed = self.decoder_embed(y)
    rnn_input = y_embed
    h0 = Variable(torch.zeros(_dims.L, _dims.N, _dims.H).type_as(rnn_input.data))
    c0 = Variable(torch.zeros(_dims.L, _dims.N, _dims.H).type_as(rnn_input.data))
    rnn_output, (ht, ct) = self.decoder_rnn(rnn_input, (h0, c0))

    rnn_output_2d = rnn_output.contiguous().view(_dims.N * _dims.T_out, _dims.H)
    output_logits = self.decoder_linear(rnn_output_2d).view(_dims.N, _dims.T_out, _dims.V_out)

    return output_logits, ht, ct

  def compute_loss(self, output_logits, y, crossent_reduction='mean'):
    """
    Compute loss. We assume that the first element of the output sequence y is
    a start token, and that each element of y is left-aligned and right-padded
    with self.NULL out to T_out. We want the output_logits to predict the
    sequence y, shifted by one timestep so that y[0] is fed to the network and
    then y[1] is predicted. We also don't want to compute loss for padded
    timesteps.

    Inputs:
    - output_logits: Variable of shape (N, T_out, V_out)
    - y: LongTensor Variable of shape (N, T_out)
    """
    # NOTE(vrama): This version of computing the loss is conceptually buggy,
    # the right log-prob of a sequence model would sum logprobs across timesteps
    # and average across mutliple datapoints in the minitbatch.
    logging.warn('Incorrect usage of log-prob in this implementation.')

    self.multinomial_outputs = None
    _dims = self.get_dims(y=y)
    mask = y.data != self.NULL
    y_mask = Variable(torch.Tensor(_dims.N, _dims.T_out).fill_(0).type_as(mask))
    y_mask[:, 1:] = mask[:, 1:]
    y_masked = y[y_mask]
    out_mask = Variable(torch.Tensor(_dims.N, _dims.T_out).fill_(0).type_as(mask))
    out_mask[:, :-1] = mask[:, 1:]
    out_mask = out_mask.view(_dims.N, _dims.T_out, 1).expand(_dims.N, _dims.T_out, _dims.V_out)
    out_masked = output_logits[out_mask].view(-1, _dims.V_out)
    # Bug in Justin's code, seems like a very common mistake.
    loss = sparse_softmax_cross_entropy_with_logits(
        logits=out_masked, labels=y_masked, reduction=crossent_reduction)
    print(torch.mean(loss))

    return loss

  def compute_loss_V2(self, output_logits, y):
    """
    Compute loss. We assume that the first element of the output sequence y is
    a start token, and that each element of y is left-aligned and right-padded
    with self.NULL out to T_out. We want the output_logits to predict the
    sequence y, shifted by one timestep so that y[0] is fed to the network and
    then y[1] is predicted. We also don't want to compute loss for padded
    timesteps.

    Inputs:
    - output_logits: Variable of shape (N, T_out, V_out)
    - y: LongTensor Variable of shape (N, T_out)
    """
    self.multinomial_outputs = None
    _dims = self.get_dims(y=y)
    mask = y.data != self.NULL

    mask_use = Variable(mask[:, 1:].float())
    y_use = y[:, 1:]
    logits_use = output_logits[:, :-1, :]

    loss = sparse_softmax_cross_entropy_with_logits(logits=logits_use,
                                                    labels=y_use)
    return torch.sum(loss*mask_use, dim=1)

  def forward(self, y, crossent_reduction='mean'):
    # TODO(vrama): Unify the code below between seq2seq.py and seq.py
    output_logits, _, _ = self.decoder(y)
    neg_logprobs = self.compute_loss_V2(output_logits, y)
    if crossent_reduction == 'mean':
      return torch.mean(neg_logprobs)
    elif crossent_reduction is None:
      return neg_logprobs
    else:
      raise NotImplementedError

