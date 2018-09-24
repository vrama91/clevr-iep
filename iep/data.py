#!/usr/bin/env python3

# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import json
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import WeightedRandomSampler

import iep.programs


def _filter_mask(arr, mask):
  if mask is not None:
    arr = arr[mask]
  return arr

def _dataset_to_tensor(dset, mask=None):
  arr = np.asarray(dset, dtype=np.int64)
  arr = _filter_mask(arr, mask)
  tensor = torch.LongTensor(arr)
  return tensor


class ClevrDataset(Dataset):
  def __init__(self, question_h5, feature_h5, vocab, mode='prefix',
               program_supervision_list=None, dont_load_features=False,
               preload_image_features=False,
               image_h5=None, max_samples=None, question_families=None,
               image_idx_start_from=None):
    mode_choices = ['prefix', 'postfix']
    if mode not in mode_choices:
      raise ValueError('Invalid mode "%s"' % mode)
    self.image_h5 = image_h5
    self.vocab = vocab
    self.feature_h5 = feature_h5
    self.dont_load_features = dont_load_features
    self.preload_image_features = preload_image_features

    if self.dont_load_features is True and self.preload_image_features is True:
      raise ValueError('Both cant be true at once.')

    # Load image features into memory.
    if self.dont_load_features:
      print('Skipping loading image features.')
      self.all_features = None
    elif self.preload_image_features:
      raise NotImplementedError
      print('Loading image features into memory.')
      all_features = np.asarray(self.feature_h5['features'], dtype=np.float32)
      self.all_features = torch.FloatTensor(all_features)

    self.mode = mode
    self.max_samples = max_samples

    mask = None
    if question_families is not None:
      # Use only the specified families
      all_families = np.asarray(question_h5['question_families'])
      N = all_families.shape[0]
      print(question_families)
      target_families = np.asarray(question_families)[:, None]
      mask = (all_families == target_families).any(axis=0)
    if image_idx_start_from is not None:
      all_image_idxs = np.asarray(question_h5['image_idxs'])
      mask = all_image_idxs >= image_idx_start_from

    # Data from the question file is small, so read it all into memory
    print('Reading question data into memory')
    self.all_questions = _dataset_to_tensor(question_h5['questions'], mask)
    self.all_image_idxs = _dataset_to_tensor(question_h5['image_idxs'], mask)
    self.all_programs = None
    if 'programs' in question_h5:
      self.all_programs = _dataset_to_tensor(question_h5['programs'], mask)
    self.all_answers = _dataset_to_tensor(question_h5['answers'], mask)

    if program_supervision_list is None:
      self.all_supervision = torch.ones(self.all_questions.size(0)).float()
    else:
      self.all_supervision = torch.from_numpy(
          program_supervision_list.astype(np.float32)).float()

    self.all_supervision = _filter_mask(self.all_supervision, mask)

    if program_supervision_list is not None and self.max_samples is not None:
      raise ValueError("Can supply either max_samples or program supervision "
                       "list at same time not both.")

  def __getitem__(self, index):
    question = self.all_questions[index]
    image_idx = self.all_image_idxs[index]
    answer = self.all_answers[index]
    supervision = self.all_supervision[index]
    program_seq = None
    if self.all_programs is not None:
      program_seq = self.all_programs[index]

    image = None
    if self.image_h5 is not None:
      image = self.image_h5['images'][image_idx]
      image = torch.FloatTensor(np.asarray(image, dtype=np.float32))

    if self.dont_load_features:
      feats = None
      #feats = self.all_features[image_idx]
    else:
      feats = self.feature_h5['features'][image_idx]
      feats = torch.FloatTensor(np.asarray(feats, dtype=np.float32))
    program_json = None
    if program_seq is not None:
      program_json_seq = []
      for fn_idx in program_seq:
        fn_str = self.vocab['program_idx_to_token'][fn_idx]
        if fn_str == '<START>' or fn_str == '<END>': continue
        fn = iep.programs.str_to_function(fn_str)
        program_json_seq.append(fn)
      if self.mode == 'prefix':
        program_json = iep.programs.prefix_to_list(program_json_seq)
      elif self.mode == 'postfix':
        program_json = iep.programs.postfix_to_list(program_json_seq)

    return (question, image, feats, answer, program_seq, program_json, supervision)

  def __len__(self):
    if self.max_samples is None:
      return self.all_questions.size(0)
    else:
      return min(self.max_samples, self.all_questions.size(0))


class ClevrDataLoader(DataLoader):
  def __init__(self, **kwargs):
    if 'question_h5' not in kwargs:
      raise ValueError('Must give question_h5')
    if 'feature_h5' not in kwargs:
      raise ValueError('Must give feature_h5')
    if 'vocab' not in kwargs:
      raise ValueError('Must give vocab')

    feature_h5_path = kwargs.pop('feature_h5')
    print('Reading features from ', feature_h5_path)
    self.feature_h5 = h5py.File(feature_h5_path, 'r')

    self.image_h5 = None
    if 'image_h5' in kwargs:
      image_h5_path = kwargs.pop('image_h5')
      print('Reading images from ', image_h5_path)
      self.image_h5 = h5py.File(image_h5_path, 'r')

    vocab = kwargs.pop('vocab')
    mode = kwargs.pop('mode', 'prefix')
    dont_load_features = kwargs.pop('dont_load_features', False)

    question_families = kwargs.pop('question_families', None)
    max_samples = kwargs.pop('max_samples', None)
    question_h5_path = kwargs.pop('question_h5')
    image_idx_start_from = kwargs.pop('image_idx_start_from', None)
    print('Reading questions from ', question_h5_path)

    with h5py.File(question_h5_path, 'r') as question_h5:
      program_supervision_npy = kwargs.pop('program_supervision_npy', None)
      if program_supervision_npy is not None:
        print('Reading program supervision file ', program_supervision_npy)
        program_supervision_list = np.load(program_supervision_npy)
        # TODO(vrama): Find a better way to do this.
        if 'bool' not in str(program_supervision_list.dtype):
          raise ValueError("Program supervision must be boolean.")
      else:
        program_supervision_list = None

      self.dataset = ClevrDataset(question_h5, self.feature_h5, vocab, mode,
                                  program_supervision_list=program_supervision_list,
                                  dont_load_features=dont_load_features,
                                  image_h5=self.image_h5,
                                  max_samples=max_samples,
                                  question_families=question_families,
                                  image_idx_start_from=image_idx_start_from)
      # Indicates which indicies in the dataset have associated supervision.
      # Use the supervision values to set the weights for sampling datapoints.
      supervision_dataset = self.dataset.all_supervision
      supervision_yes = torch.sum(supervision_dataset == 1.0)
      supervision_no = torch.sum(supervision_dataset == 0.0)
      mixing_factor = kwargs.pop('mixing_factor_supervision', 1)
      if mixing_factor > 1 or mixing_factor < 0:
        raise ValueError("Mixing factor is bounded above by 1, below by 0.")
      if supervision_no != 0:
        weights_sampling_dataset = torch.zeros(supervision_dataset.size()).double()
        weights_sampling_dataset[
            supervision_dataset == 1.0] = mixing_factor/supervision_yes
        weights_sampling_dataset[
            supervision_dataset == 0.0] = 1/supervision_no
        weighted_random_sampler = WeightedRandomSampler(
            weights=weights_sampling_dataset, num_samples=len(self.dataset),
            replacement=True)
        print('Using a weighted random sampler.')
        kwargs['sampler'] = weighted_random_sampler
        print('Forcing shuffle to be false as we are semi-supervised.')
        kwargs['shuffle'] = False
      elif supervision_yes == 0:
        raise RuntimeError("Cannot work with 0 supervision.")

    kwargs['collate_fn'] = clevr_collate
    super(ClevrDataLoader, self).__init__(self.dataset, **kwargs)

  def close(self):
    if self.image_h5 is not None:
      self.image_h5.close()
    if self.feature_h5 is not None:
      self.feature_h5.close()

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    self.close()


def clevr_collate(batch):
  transposed = list(zip(*batch))
  question_batch = default_collate(transposed[0])
  image_batch = transposed[1]
  if any(img is not None for img in image_batch):
    image_batch = default_collate(image_batch)
  feat_batch = transposed[2]
  if any(f is not None for f in feat_batch):
    feat_batch = default_collate(feat_batch)
  answer_batch = default_collate(transposed[3])
  program_seq_batch = transposed[4]
  if transposed[4][0] is not None:
    program_seq_batch = default_collate(transposed[4])
  program_struct_batch = transposed[5]
  supervision_batch = default_collate(transposed[6])
  return [question_batch, image_batch, feat_batch, answer_batch,
          program_seq_batch, program_struct_batch, supervision_batch]
