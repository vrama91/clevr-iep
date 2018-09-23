#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 rvedantam3 <vrama91@vt.edu>
#
# Distributed under terms of the MIT license.

"""
A script to create a list of datapoints for which we have supervision on CLEVR.
"""
import argparse
import h5py
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('--input_questions_h5', default=None)
parser.add_argument('--num_supervised', default=5000, type=int)
parser.add_argument('--output_npy_file', default=None)
parser.add_argument('--random_seed', default=42, type=int)

args = parser.parse_args()

if args.input_questions_h5 is None or args.output_npy_file is None:
  raise ValueError("Must provide all arguments.")

if "train" not in args.input_questions_h5:
  raise ValueError

dataset = h5py.File(args.input_questions_h5, 'r')
num_questions = dataset['questions'].shape[0]

print('Number of supervised data points: %d, size of the dataset: %d' %
      (args.num_supervised, num_questions))

print('Setting np random seed')
np.random.seed(args.random_seed)
supervision_array = np.zeros(num_questions).astype(np.bool)
supervised_points = np.random.choice(list(range(num_questions)), replace=False, size=args.num_supervised)
supervision_array[supervised_points] = True

print('Saving supervision array to disk')
np.save(args.output_npy_file, supervision_array)
