#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 rvedantam3 <vrama91@vt.edu>
#
# Distributed under terms of the MIT license.

"""

"""
import torch
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from collections import namedtuple


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
    self.decoder_rnn = nn.LSTM(wordvec_dim + hidden_dim, hidden_dim, rnn_num_layers,
                               dropout=rnn_dropout, batch_first=True)
    self.decoder_linear = nn.Linear(hidden_dim, decoder_vocab_size)
    self.DECODER_INIT = torch.Tensor(torch.randn(hidden_dim))
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

  def decoder(self, decoder_init, y, h0=None, c0=None):
    _dims = self.get_dims(y=y)

    if _dims.T_out > 1:
      y, _ = self.before_rnn(y)
    y_embed = self.decoder_embed(y)
    decoder_init_repeat = decoder_init.view(_dims.N, 1, _dims.H).expand(_dims.N, _dims.T_out, _dims.H)
    rnn_input = torch.cat([decoder_init_repeat, y_embed], 2)
    h0 = Variable(torch.zeros(_dims.L, _dims.N, _dims.H).type_as(decoder_init))
    c0 = Variable(torch.zeros(_dims.L, _dims.N, _dims.H).type_as(decoder_init))
    rnn_output, (ht, ct) = self.decoder_rnn(rnn_input, (h0, c0))

    rnn_output_2d = rnn_output.contiguous().view(_dims.N * _dims.T_out, _dims.H)
    output_logprobs = self.decoder_linear(rnn_output_2d).view(_dims.N, _dims.T_out, _dims.V_out)

    return output_logprobs, ht, ct

  def compute_loss(self, output_logprobs, y):
    """
    Compute loss. We assume that the first element of the output sequence y is
    a start token, and that each element of y is left-aligned and right-padded
    with self.NULL out to T_out. We want the output_logprobs to predict the
    sequence y, shifted by one timestep so that y[0] is fed to the network and
    then y[1] is predicted. We also don't want to compute loss for padded
    timesteps.

    Inputs:
    - output_logprobs: Variable of shape (N, T_out, V_out)
    - y: LongTensor Variable of shape (N, T_out)
    """
    self.multinomial_outputs = None
    _dims = self.get_dims(y=y)
    mask = y.data != self.NULL
    y_mask = Variable(torch.Tensor(_dims.N, _dims.T_out).fill_(0).type_as(mask))
    y_mask[:, 1:] = mask[:, 1:]
    y_masked = y[y_mask]
    out_mask = Variable(torch.Tensor(_dims.N, _dims.T_out).fill_(0).type_as(mask))
    out_mask[:, :-1] = mask[:, 1:]
    out_mask = out_mask.view(_dims.N, _dims.T_out, 1).expand(_dims.N, _dims.T_out, _dims.V_out)
    out_masked = output_logprobs[out_mask].view(-1, _dims.V_out)
    loss = F.cross_entropy(out_masked, y_masked)
    return loss

  def forward(self, y):
    _dims = self.get_dims(y=y)
    decoder_init = self.DECODER_INIT.repeat(_dims.N, 1)
    output_logprobs, _, _ = self.decoder(decoder_init, y)
    loss = self.compute_loss(output_logprobs, y)
    return loss
