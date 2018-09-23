#!/usr/bin/env python3

# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
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


def invert_dict(d):
  return {v: k for k, v in d.items()}