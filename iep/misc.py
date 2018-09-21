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

  log_probs = log_softmax(logits_flat)
  negative_log_likelihood = -1 * torch.gather(
      log_probs, 1, labels_flat.unsqueeze(1))

  if logits.dim() == 3:
    negative_log_likelihood = negative_log_likelihood.view(size_0, size_1)

  return negative_log_likelihood
