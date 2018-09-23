# Script to evaluate models.
import numpy as np
import torch
import iep.utils as utils
import iep.preprocess

from torch.autograd import Variable

# TODO(vrama): Write this piece of code.
from checkpoints import latest_checkpoint
from iep.misc import set_mode

def check_accuracy(args, program_prior, question_reconstructor,
                   program_generator, execution_engine, baseline_model, loader):
  set_mode('eval', [
      program_prior, question_reconstructor, program_generator,
      execution_engine, baseline_model
  ])
  num_correct, num_samples = 0, 0
  for batch in loader:
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
      vocab = utils.load_vocab(args.vocab_json)
      for i in range(questions.size(0)):
        program_pred = program_generator.sample(
            Variable(questions[i:i + 1].cuda(), volatile=True))
        program_pred_str = iep.preprocess.decode(program_pred,
                                                 vocab['program_idx_to_token'])
        program_str = iep.preprocess.decode(programs[i],
                                            vocab['program_idx_to_token'])
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