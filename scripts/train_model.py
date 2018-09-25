#!/usr/bin/env python3

# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
import os
import gc

import argparse
import copy
import json
import random
import shutil

import torch
torch.backends.cudnn.enabled = True
from torch.autograd import Variable
import torch.nn.functional as F
import time
import numpy as np
import h5py

import iep.utils as utils
import iep.preprocess

from tensorboardX import SummaryWriter
from iep.data import ClevrDataLoader
from iep.models import ModuleNet
from iep.models import Seq2Seq
from iep.models import LstmModel
from iep.models import CnnLstmModel
from iep.models import CnnLstmSaModel
from iep.models.seq import SeqModel

from iep.misc import check_accuracy
from iep.misc import apply_supervision_on_loss
from iep.misc import get_baseline_model
from iep.misc import get_execution_engine
from iep.misc import get_program_generator
from iep.misc import get_program_prior
from iep.misc import get_question_reconstructor
from iep.misc import get_state
from iep.misc import set_mode
from iep.misc import load_vocab

parser = argparse.ArgumentParser()

_TRAIN_DATA_DIR = '/srv/share/datasets/clevr/CLEVR_v1.0/clevr-iep-data/'

# Input data
parser.add_argument(
    '--train_question_h5', default=_TRAIN_DATA_DIR + 'train_questions.h5')
parser.add_argument(
    '--train_features_h5', default=_TRAIN_DATA_DIR + 'train_features.h5')
parser.add_argument(
    '--val_question_h5', default=_TRAIN_DATA_DIR + 'val_questions.h5')
parser.add_argument(
    '--val_features_h5', default=_TRAIN_DATA_DIR + 'val_features.h5')

# Split to use for evaluation mode of the script.
parser.add_argument(
    '--only_evaluation_split',
    default='',
    help=
    'If set to anything other than empty, then skips training and runs '
    'evaluation on the specified split'
)
parser.add_argument('--feature_dim', default='1024,14,14')
parser.add_argument('--vocab_json', default=_TRAIN_DATA_DIR + 'vocab.json')
parser.add_argument(
    '--dont_load_train_features_memory', default=False, action='store_true')
parser.add_argument(
    '--preload_image_features_ram', default=False, action='store_true')

parser.add_argument('--loader_num_workers', type=int, default=1)
parser.add_argument('--evaluate_during_train', action='store_true', default=False)
parser.add_argument('--use_local_copies', default=0, type=int)
parser.add_argument('--cleanup_local_copies', default=1, type=int)

parser.add_argument('--family_split_file', default=None)
parser.add_argument('--num_train_samples', default=None, type=int)
parser.add_argument('--num_val_samples', default=10000, type=int)
parser.add_argument('--shuffle_train_data', default=1, type=int)

parser.add_argument('--program_supervision_npy', default=None, type=str)
parser.add_argument('--mixing_factor_supervision', default=1.0, type=float)
parser.add_argument('--max_program_length', default=30, type=int)
parser.add_argument('--max_question_length', default=50, type=int)

# What type of model to use and which parts to train
parser.add_argument('--model_version', default='iep', help='can be either of iep or discovery')
parser.add_argument(
    '--model_type',
    default='PG',
    choices=['Prior', 'PG', 'EE', 'PG+EE', 'LSTM', 'CNN+LSTM', 'CNN+LSTM+SA'])
parser.add_argument('--train_program_generator', default=1, type=int)
parser.add_argument('--train_execution_engine', default=1, type=int)
parser.add_argument('--baseline_train_only_rnn', default=0, type=int)

parser.add_argument('--use_gt_programs_for_ee', default=0, type=int)

# Discovery model options.
parser.add_argument('--discovery_alpha', default=100.0, type=float,)
parser.add_argument('--discovery_beta', default=0.1, type=float,)
parser.add_argument('--discovery_gamma', default=1.0, type=float,)

# Start from an existing checkpoint
parser.add_argument('--program_prior_start_from', default=None)
parser.add_argument('--question_reconstructor_start_from', default=None)
parser.add_argument('--program_generator_start_from', default=None)
parser.add_argument('--execution_engine_start_from', default=None)
parser.add_argument('--baseline_start_from', default=None)

# RNN options
parser.add_argument('--rnn_wordvec_dim', default=300, type=int)
parser.add_argument('--rnn_hidden_dim', default=256, type=int)
parser.add_argument('--rnn_num_layers', default=2, type=int)
parser.add_argument('--rnn_dropout', default=0, type=float)

# Module net options
parser.add_argument('--module_stem_num_layers', default=2, type=int)
parser.add_argument('--module_stem_batchnorm', default=0, type=int)
parser.add_argument('--module_dim', default=128, type=int)
parser.add_argument('--module_residual', default=1, type=int)
parser.add_argument('--module_batchnorm', default=0, type=int)

# CNN options (for baselines)
parser.add_argument('--cnn_res_block_dim', default=128, type=int)
parser.add_argument('--cnn_num_res_blocks', default=0, type=int)
parser.add_argument('--cnn_proj_dim', default=512, type=int)
parser.add_argument(
    '--cnn_pooling', default='maxpool2', choices=['none', 'maxpool2'])

# Stacked-Attention options
parser.add_argument('--stacked_attn_dim', default=512, type=int)
parser.add_argument('--num_stacked_attn', default=2, type=int)

# Classifier options
parser.add_argument('--classifier_proj_dim', default=512, type=int)
parser.add_argument(
    '--classifier_downsample',
    default='maxpool2',
    choices=['maxpool2', 'maxpool4', 'none'])
parser.add_argument('--classifier_fc_dims', default='1024')
parser.add_argument('--classifier_batchnorm', default=0, type=int)
parser.add_argument('--classifier_dropout', default=0, type=float)

# Optimization options
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--num_iterations', default=100000, type=int)
parser.add_argument('--learning_rate', default=5e-4, type=float)
parser.add_argument('--reward_decay', default=0.9, type=float)
parser.add_argument('--elbo_reward_decay', default=0.99, type=float)

# Output options
parser.add_argument('--checkpoint_dir', default='data/')
parser.add_argument('--randomize_checkpoint_dir', type=int, default=0)
parser.add_argument('--record_loss_every', type=int, default=1)
parser.add_argument('--checkpoint_every', default=10000, type=int)

def _global_step_from_checkpoint(checkpoint_path):
  global_step = int(
    checkpoint_path.split('/')[-1].split('-')[-1].split('.')[0])
  return global_step

def main(args):
  if args.randomize_checkpoint_dir == 1:
    name, ext = os.path.splitext(args.checkpoint_dir)
    num = random.randint(1, 1000000)
    args.checkpoint_dir = '%s_%06d%s' % (name, num, ext)

  vocab = load_vocab(args.vocab_json)

  if args.use_local_copies == 1:
    shutil.copy(args.train_question_h5, '/tmp/train_questions.h5')
    shutil.copy(args.train_features_h5, '/tmp/train_features.h5')
    shutil.copy(args.val_question_h5, '/tmp/val_questions.h5')
    shutil.copy(args.val_features_h5, '/tmp/val_features.h5')
    args.train_question_h5 = '/tmp/train_questions.h5'
    args.train_features_h5 = '/tmp/train_features.h5'
    args.val_question_h5 = '/tmp/val_questions.h5'
    args.val_features_h5 = '/tmp/val_features.h5'

  question_families = None
  if args.family_split_file is not None:
    with open(args.family_split_file, 'r') as f:
      question_families = json.load(f)

  train_loader_kwargs = {
      'question_h5': args.train_question_h5,
      'feature_h5': args.train_features_h5,
      'vocab': vocab,
      'batch_size': args.batch_size,
      'shuffle': args.shuffle_train_data == 1,
      'question_families': question_families,
      'max_samples': args.num_train_samples,
      'num_workers': args.loader_num_workers,
      'program_supervision_npy': args.program_supervision_npy,
      'mixing_factor_supervision': args.mixing_factor_supervision,
      'dont_load_features': args.dont_load_train_features_memory,
      'preload_image_features': args.preload_image_features_ram,
  }
  val_loader_kwargs = {
      'question_h5': args.val_question_h5,
      'feature_h5': args.val_features_h5,
      'vocab': vocab,
      'shuffle': False,
      'batch_size': args.batch_size,
      'question_families': question_families,
      'max_samples': args.num_val_samples,
      'num_workers': args.loader_num_workers,
      'preload_image_features': args.preload_image_features_ram,
  }

  if args.only_evaluation_split != '':
    with ClevrDataLoader(**val_loader_kwargs) as val_loader:
      print('Evaluation only, no training')
      if args.num_val_samples > 10000:
        raise ValueError('First 10000 samples only for validation')
      eval_loop(args, val_loader, split_name=args.only_evaluation_split)
  else:
    with ClevrDataLoader(**train_loader_kwargs) as train_loader, \
         ClevrDataLoader(**val_loader_kwargs) as val_loader:
      print('Training')
      train_loop(args, train_loader, val_loader)

  if args.use_local_copies == 1 and args.cleanup_local_copies == 1:
    os.remove('/tmp/train_questions.h5')
    os.remove('/tmp/train_features.h5')
    os.remove('/tmp/val_questions.h5')
    os.remove('/tmp/val_features.h5')


def eval_loop(args, loader, split_name='val', checkpoint_prefix='model'):
  writer = SummaryWriter(log_dir=args.checkpoint_dir)
  candidate_checkpoints = [
    os.path.join(args.checkpoint_dir, x) for x in os.listdir(args.checkpoint_dir) if (
      checkpoint_prefix in x and x.split('.')[-1]=='ckpt')]
  if len(candidate_checkpoints) == 0:
    print('No checkpoints found, quitting.')
    return
  else:
    print('Evaluating %d checkpoints' % (len(candidate_checkpoints)))

  vocab = load_vocab(args.vocab_json)
  print('loader has %d samples' % len(loader.dataset))
  best_accuracy = -100000
  best_ckpt = None

  for idx_checkpoint, checkpoint in enumerate(candidate_checkpoints):
    global_step = _global_step_from_checkpoint(checkpoint)

    # load the checkpoint and evaluate the model.
    program_prior = None
    question_reconstructor = None
    program_generator = None
    execution_engine = None
    baseline_model = None

    # Set up model
    if args.model_type == 'Prior':
      args.program_prior_start_from = checkpoint
      program_prior, prior_kwargs = get_program_prior(args)
      print('Here is the program prior:')
      print(program_prior)
    if args.model_type == 'PG' or args.model_type == 'PG+EE':
      args.program_generator_start_from = checkpoint
      program_generator, pg_kwargs = get_program_generator(args)
      print('Here is the program generator:')
      print(program_generator)
      if args.model_version == 'discovery':
        args.question_reconstructor_start_from = checkpoint
        question_reconstructor, qr_kwargs = get_question_reconstructor(args)
        print('Here is the question reconstructor:')
        print(question_reconstructor)

        args.program_prior_start_from = checkpoint
        program_prior, prior_kwargs = get_program_prior(args)
        print('Here is the program prior:')
        print(program_prior)
    if args.model_type == 'EE' or args.model_type == 'PG+EE':
      args.program_generator_start_from = checkpoint
      program_generator, pg_kwargs = get_program_generator(args)
      print('Here is the program generator:')
      print(program_generator)

      execution_engine, ee_kwargs = get_execution_engine(args)
      print('Here is the execution engine:')
      print(execution_engine)
    if args.model_type in ['LSTM', 'CNN+LSTM', 'CNN+LSTM+SA']:
      args.baseline_start_from = checkpoint
      baseline_model, baseline_kwargs = get_baseline_model(args)
      params = baseline_model.parameters()
      if args.baseline_train_only_rnn == 1:
        params = baseline_model.rnn.parameters()
      print('Here is the baseline model')
      print(baseline_model)
      baseline_type = args.model_type

    set_mode('eval', [
      program_prior, question_reconstructor, program_generator,
      execution_engine, baseline_model
    ])

    print('Checking validation accuracy ...')
    stime = time.time()
    accuracy = check_accuracy(args, program_prior, question_reconstructor, program_generator,
                             execution_engine, baseline_model, loader)
    writer.add_scalar('data/' + split_name + '_accuracy', accuracy, global_step)
    print('Ckpt[%d/%d] %s accuracy at step %d is %f, eval took %f sec. ' %(idx_checkpoint, len(candidate_checkpoints),
      split_name, global_step, accuracy, time.time()-stime))

    if accuracy > best_accuracy:
      best_accuracy = accuracy
      print('Found best checkpoint at %d step' %(global_step))
      best_ckpt = {'checkpoint': checkpoint, 'accuracy': accuracy}

  writer.export_scalars_to_json(os.path.join(args.checkpoint_dir, split_name + 'all_evaluation_scalars.json'))
  with open(os.path.join(args.checkpoint_dir, split_name + 'best_checkpoint.json'), 'w') as f:
    print(best_ckpt)
    json.dump(best_ckpt, f)
  writer.close()


def train_loop(args, train_loader, val_loader):
  print('Using log directory', args.checkpoint_dir)
  writer = SummaryWriter(log_dir=args.checkpoint_dir)

  vocab = load_vocab(args.vocab_json)

  program_prior, prior_kwargs, prior_optimizer = None, None, None
  question_reconstructor, qr_kwargs, qr_optimizer = None, None, None
  program_generator, pg_kwargs, pg_optimizer = None, None, None
  execution_engine, ee_kwargs, ee_optimizer = None, None, None
  baseline_model, baseline_kwargs, baseline_optimizer = None, None, None
  baseline_type = None

  # Set up model
  if args.model_type == 'Prior':
    program_prior, prior_kwargs = get_program_prior(args)
    prior_optimizer = torch.optim.Adam(
        program_prior.parameters(), lr=args.learning_rate)
    print('Here is the program prior:')
    print(program_prior)
  if args.model_type == 'PG' or args.model_type == 'PG+EE':
    program_generator, pg_kwargs = get_program_generator(args)
    pg_optimizer = torch.optim.Adam(
        program_generator.parameters(), lr=args.learning_rate)
    print('Here is the program generator:')
    print(program_generator)
    if args.model_version == 'discovery':
      question_reconstructor, qr_kwargs = get_question_reconstructor(args)
      qr_optimizer = torch.optim.Adam(
          question_reconstructor.parameters(), lr=args.learning_rate)
      print('Here is the question reconstructor:')
      print(question_reconstructor)
      program_prior, prior_kwargs = get_program_prior(args)
      prior_optimizer = torch.optim.Adam(
          program_prior.parameters(), lr=args.learning_rate)
      print('Here is the program prior:')
      print(program_prior)
  if args.model_type == 'EE': 
    program_generator, pg_kwargs = get_program_generator(args)
    pg_optimizer = torch.optim.Adam(
        program_generator.parameters(), lr=args.learning_rate)
    print('Here is the program generator:')
    print(program_generator)

    execution_engine, ee_kwargs = get_execution_engine(args)
    ee_optimizer = torch.optim.Adam(
        execution_engine.parameters(), lr=args.learning_rate)
    print('Here is the execution engine:')
    print(execution_engine)
  if args.model_type == 'PG+EE':
    program_prior, prior_kwargs = get_program_prior(args)
    print('Here is the program prior:')
    print(program_prior)

    question_reconstructor, qr_kwargs = get_question_reconstructor(args)
    qr_optimizer = torch.optim.Adam(
        question_reconstructor.parameters(), lr=args.learning_rate)
    print('Here is the question reconstructor:')
    print(question_reconstructor)

    program_generator, pg_kwargs = get_program_generator(args)
    pg_optimizer = torch.optim.Adam(
        program_generator.parameters(), lr=args.learning_rate)
    print('Here is the program generator:')
    print(program_generator)

    execution_engine, ee_kwargs = get_execution_engine(args)
    ee_optimizer = torch.optim.Adam(
        execution_engine.parameters(), lr=args.learning_rate)
    print('Here is the execution engine:')
    print(execution_engine)

  if args.model_type in ['LSTM', 'CNN+LSTM', 'CNN+LSTM+SA']:
    baseline_model, baseline_kwargs = get_baseline_model(args)
    params = baseline_model.parameters()
    if args.baseline_train_only_rnn == 1:
      params = baseline_model.rnn.parameters()
    baseline_optimizer = torch.optim.Adam(params, lr=args.learning_rate)
    print('Here is the baseline model')
    print(baseline_model)
    baseline_type = args.model_type
  loss_fn = torch.nn.CrossEntropyLoss().cuda()

  stats = {
      'train_losses': [],
      'train_rewards': [],
      'train_losses_ts': [],
      'train_accs': [],
      'val_accs': [],
      'val_accs_ts': [],
      'best_val_acc': -100000,
      'model_t': 0,
  }
  t, epoch, reward_moving_average = 0, 0, 0

  set_mode('train', [
      program_prior, question_reconstructor, program_generator,
      execution_engine, baseline_model
  ])
  print('train_loader has %d samples' % len(train_loader.dataset))
  print('val_loader has %d samples' % len(val_loader.dataset))

  while t < args.num_iterations:
    epoch += 1
    print('Starting epoch %d' % epoch)
    for batch in train_loader:
      t += 1
      if t == 1:
        stime = time.time()
      questions, _, feats, answers, programs, _, supervision = batch
      questions_var = Variable(questions.cuda())

      if feats[0] is not None:
        feats_var = Variable(feats.cuda())

      answers_var = Variable(answers.cuda())
      supervision_var = Variable(supervision.cuda())

      if programs[0] is not None:
        programs_var = Variable(programs.cuda())

      reward = None
      if args.model_type == 'Prior':
        prior_optimizer.zero_grad()
        loss = program_prior(programs_var)
        loss.backward()
        print_loss = loss
        prior_optimizer.step()
      elif args.model_type == 'PG':
        # Train program generator with ground-truth programs
        pg_optimizer.zero_grad()
        supervised_loss = program_generator(
            questions_var, programs_var, crossent_reduction=None)
        supervised_loss = apply_supervision_on_loss(
          supervised_loss, supervision_var)
        supervised_loss = supervised_loss * args.discovery_alpha
        print_loss = supervised_loss

        if args.model_version == 'iep':
          supervised_loss.backward(retain_graph=True)
          pg_optimizer.step()
        elif args.model_version == 'discovery':
          # Setup for discovery models.
          all_discovery_losses = supervised_loss
          toggle_program_prior = False
          if program_prior.training:
            toggle_program_prior = True
            set_mode('eval', [program_prior])
          qr_optimizer.zero_grad()
          sampled_program = program_generator.reinforce_sample(questions_var)

          # Compute logprobs that we need.
          neg_logprob_program_generator = program_generator(
              questions_var, sampled_program, crossent_reduction=None)
          neg_logprob_reconstruction = question_reconstructor(
              sampled_program, questions_var, crossent_reduction=None)
          neg_logprob_prior = program_prior(sampled_program, crossent_reduction=None)

          # Question reconstruction: backward calls and gradient updates.
          question_reconstruction_loss = torch.mean(neg_logprob_reconstruction)
          path_derivative_loss_program_generator = -1 * args.discovery_beta * torch.mean(
              neg_logprob_program_generator)
          all_discovery_losses = all_discovery_losses + path_derivative_loss_program_generator
          all_discovery_losses = all_discovery_losses + question_reconstruction_loss

          # NELBO loss: sets up the reinforce update.
          nelbo_loss = neg_logprob_reconstruction + args.discovery_beta * (
            neg_logprob_prior - neg_logprob_program_generator)
          reward_elbo = -1 * nelbo_loss.float()
          reward_moving_average = reward_moving_average * args.elbo_reward_decay
          reward_moving_average = reward_moving_average + (
            1.0 - args.elbo_reward_decay) * reward_elbo.data.mean()
          centered_elbo_reward = reward_elbo - reward_moving_average
          reinforce_loss = torch.mean(
            neg_logprob_program_generator * centered_elbo_reward.detach())

          all_discovery_losses += reinforce_loss

          # Above loss is for all the gradients we need to compute in the model, print
          # loss stores what makes sense to look at as an objective for training.
          print_loss = print_loss + torch.mean(nelbo_loss)
          writer.add_scalar('data/nelbo_loss', torch.mean(nelbo_loss).data[0], global_step=t)
          print('Step: %d, NELBO loss: %f'% (t, torch.mean(nelbo_loss).data[0]))

          all_discovery_losses.backward()
          qr_optimizer.step()
          pg_optimizer.step()

          if toggle_program_prior:
            set_mode('train', [program_prior])

      elif args.model_type == 'EE':
        ee_optimizer.zero_grad()

        if args.use_gt_programs_for_ee == 1:
          programs_to_use = programs_var
        elif args.use_gt_programs_for_ee == 0:
          programs_to_use = program_generator.reinforce_sample(
            questions_var, argmax=True)

        scores = execution_engine(feats_var, programs_to_use)
        loss = loss_fn(scores, answers_var)

        scores_gt = execution_engine(feats_var, programs_var)
        loss_gt = loss_fn(scores_gt, answers_var)

        print('loss', loss, 'loss_gt', loss_gt)

        loss.backward()
        print_loss = loss
        ee_optimizer.step()
      elif args.model_type in ['LSTM', 'CNN+LSTM', 'CNN+LSTM+SA']:
        baseline_optimizer.zero_grad()
        baseline_model.zero_grad()
        scores = baseline_model(questions_var, feats_var)
        loss = loss_fn(scores, answers_var)
        loss.backward()
        print_loss = loss
        baseline_optimizer.step()
      elif args.model_type == 'PG+EE':
        raise NotImplementedError
        programs_pred = program_generator.reinforce_sample(questions_var)
        scores = execution_engine(feats_var, programs_pred)

        loss = loss_fn(scores, answers_var)
        _, preds = scores.data.cpu().max(1)

        # Will need to change the raw reward computation here to include
        # the other stuff.
        raw_reward = (preds == answers).float()
        reward_moving_average *= args.reward_decay
        reward_moving_average += (1.0 - args.reward_decay) * raw_reward.mean()
        centered_reward = raw_reward - reward_moving_average

        if args.train_execution_engine == 1:
          ee_optimizer.zero_grad()
          loss.backward()
          ee_optimizer.step()

        if args.train_program_generator == 1:
          pg_optimizer.zero_grad()
          program_generator.reinforce_backward(centered_reward.cuda())
          pg_optimizer.step()
        print_loss = loss

      writer.add_scalar('data/train_loss', print_loss.data[0], t)
      if t != 1:
        writer.add_scalar('runtime/sec_per_step', (time.time() - stime)/(t-1), t)
      writer.add_scalar('data/supervision',
                        float(torch.sum(supervision_var).data[0]) / args.batch_size, t)

      if t % args.record_loss_every == 0 and t != 1:
        print('Step: %d, loss: %f (%f Sec./step) [%f supervision]' %
              (t, float(print_loss.data[0]), (time.time() - stime) / t - 1,
               float(torch.sum(supervision_var).data[0]) / args.batch_size))
        stats['train_losses'].append(print_loss.data[0])
        stats['train_losses_ts'].append(t)
        if reward is not None:
          stats['train_rewards'].append(reward)

      if t % args.checkpoint_every == 0:
        if args.model_type == 'EE' and program_generator is None:
          raise ValueError

        if args.evaluate_during_train:
          print('Checking validation accuracy ...')
          val_acc = check_accuracy(args, program_prior, question_reconstructor, program_generator,
                                   execution_engine, baseline_model, val_loader)
          writer.add_scalar('data/val_accuracy', val_acc, t)
          print('val accuracy is ', val_acc)
          stats['val_accs'].append(val_acc)
          stats['val_accs_ts'].append(t)
          if val_acc > stats['best_val_acc']:
            stats['best_val_acc'] = val_acc
            stats['model_t'] = t

        prior_state = get_state(program_prior)
        qr_state = get_state(question_reconstructor)
        pg_state = get_state(program_generator)
        ee_state = get_state(execution_engine)
        baseline_state = get_state(baseline_model)
        checkpoint = {
            'args': args.__dict__,
            'program_prior_kwargs': prior_kwargs,
            'program_prior_state': prior_state,
            'question_reconstructor_kwargs': qr_kwargs,
            'question_reconstructor_state': qr_state,
            'program_generator_kwargs': pg_kwargs,
            'program_generator_state': pg_state,
            'execution_engine_kwargs': ee_kwargs,
            'execution_engine_state': ee_state,
            'baseline_kwargs': baseline_kwargs,
            'baseline_state': baseline_state,
            'baseline_type': baseline_type,
            'vocab': vocab
        }
        ckpt_path = os.path.join(args.checkpoint_dir, 'model-%d.ckpt' % (t))
        print('Saving checkpoint to %s' % ckpt_path)

        torch.save(checkpoint, ckpt_path)
        del checkpoint['program_prior_state']
        del checkpoint['question_reconstructor_state']
        del checkpoint['program_generator_state']
        del checkpoint['execution_engine_state']
        del checkpoint['baseline_state']
        with open(os.path.join(args.checkpoint_dir, 'metadata-%d.json' % (t)), 'w') as f:
          json.dump(checkpoint, f)

      if t == args.num_iterations:
        writer.export_scalars_to_json(os.path.join(args.checkpoint_dir, 'all_scalars.json'))
        writer.close()
        break


if __name__ == '__main__':
  _HOSTNAME = os.environ.get("HOSTNAME")
  print("Running experiment on %s." % (_HOSTNAME))
  args = parser.parse_args()

  print("Model is set to %s type and %s version." % (args.model_type, args.model_version))
  main(args)