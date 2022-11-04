import os
from os.path import join, abspath, dirname
import uuid
import random
from pickletools import optimize
import argparse

import torch
import numpy as np

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

from tqdm import trange, tqdm

import log
from models import GenArgs, GenModel

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="6"
os.environ['TRANSFORMERS_CACHE'] = 'xxx'

def extract_sentence(sample, tokenizer):
    input_ids = sample["input_ids"]
    print('--- ans',sample["tgt_text"])
    for tokens in input_ids:
        tokens = tokens.cpu().tolist()
        clean = [x for x in tokens if x != 0]
        sent = tokenizer.decode(clean)
        print('----', sent)


if __name__ == '__main__':
    gen_args = GenArgs(
            task_name='wic',
            data_dir='../FewGLUE_32dev/WiC',
            template='templates/wic.gen.txt',
            labels='templates/wic.multi-labels.txt',
            model_name='t5-lm',
            model_path='google/t5-base-lm-adapt',
            max_seq_len=480,
            train_batch_size=2,
            eval_batch_size=1,
            use_cuda=True
        )
    model = GenModel(gen_args)

    cnt = 0
    for sample in model.valid_dataloader:
        extract_sentence(sample, model.tokenizer)
        out, sentences = model.forward(sample.cuda())
        print(out, sentences)
        cnt += 1
        if cnt > 5:
            break
