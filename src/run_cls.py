import os
from os.path import join, abspath, dirname
import uuid
import random
from pickletools import optimize
import argparse

import torch
import numpy as np

from openprompt import PromptForClassification, PromptDataLoader
from openprompt.prompts import ManualVerbalizer, PtuningTemplate
from openprompt.data_utils.utils import InputExample
from openprompt.plms import load_plm

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

from tqdm import trange, tqdm

import log
from advml_lib import SynsetVerbalizer
from models import PtuningModel, PtuningArgs, DirectCmpModel, PositionAwareModel


SUPPORT_MODELS = ['bert-base-cased', 'bert-large-cased',
                  'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl',
                  'roberta-base', 'roberta-large',
                  'albert-xxlarge-v2',
                  'megatron_11b']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='configs/default.cfg')

    # general
    parser.add_argument("--task_name", type=str, default='wic')

    # train config
    parser.add_argument("--model_name", type=str, default="albert")
    parser.add_argument("--model_path", type=str, default="albert-xxlarge-v2", choices=SUPPORT_MODELS)
    parser.add_argument("--seed", type=int, default=200)
    parser.add_argument("--template", type=str, help="prompt template")
    parser.add_argument("--labels", type=str, help="prompt template")
    
    parser.add_argument("--use_cuda", type=bool, default=True)

    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=3500)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=0)
    
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=1)


    # directories
    parser.add_argument("--data_dir", type=str, default='../FewGLUE_32dev/WiC', help="dataset path")
    parser.add_argument("--out_dir", type=str, default='../../out/ckpts', help="model ckpt output")
    parser.add_argument("--save_model", type=bool, default=True)

    # log
    parser.add_argument("--eval_every_step", type=int, default=20)

    # debug
    parser.add_argument("--debug_step_interval", type=int, default=5)
    parser.add_argument("--debug", type=bool, default=True)


    args = parser.parse_args()
    return args


logger = log.get_logger("train")
class Trainer:

    def __init__(self, args: argparse.ArgumentParser):
        self.args = args
        #self.set_seed(self.args.seed)
        the_seed = random.randint(0, 10000)
        self.set_seed(the_seed)
        # pt_args = PtuningArgs.fromFile("configs/twoword.cfg")
        # pt_args = PtuningArgs.fromFile("configs/default.cfg")
        print(f"config: {args.config}")
        pt_args = PtuningArgs.fromFile(args.config)
        pt_args.template = args.template
        
        # self.model = PtuningModel(pt_args)
        self.model = DirectCmpModel(pt_args)
        # self.model = PositionAwareModel(pt_args)
        self.train_dataloader = self.model.train_dataloader
        self.valid_dataloader = self.model.valid_dataloader
        self.test_dataloader = self.model.test_dataloader
        self.prompt_model = self.model.prompt_model

        # prepare model id, for model saving
        from datetime import datetime
        ts = datetime.today().strftime('%Y-%m-%d-%H-%M')
        self.id = f"{ts}_{uuid.uuid4()}"
        print(f"job id is {self.id} random seed {the_seed}")


    def set_seed(self, seed):
        seed = int(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


    def eval(self, dataloader):
        allpreds = []
        alllabels = []
        
        iterator = tqdm(dataloader, desc="eval")
        for step, inputs in enumerate(iterator):
        # for step, inputs in enumerate(dataloader):
            if self.args.use_cuda:
                inputs = inputs.cuda()
            logits = self.model.forward(inputs)
            labels = inputs['label']
            alllabels.extend(labels.cpu().tolist())
            allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
        
        np_allpreds = np.asarray(allpreds)
        one = np_allpreds.sum() / len(allpreds)

        acc = sum([int(i==j) for i,j in zip(allpreds, alllabels)])/len(allpreds)
        return acc, one


    def train(self):
        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            num_train_epochs = self.args.max_steps // (max(1, len(self.train_dataloader) // self.args.gradient_accumulation_steps)) + 1
        else:
            t_total = len(self.train_dataloader) // self.gradient_accumulation_steps * self.args.num_train_epochs

        print("\n")
        print("num_steps_per_dataset:")
        print(len(self.train_dataloader) // self.args.gradient_accumulation_steps)
        print("total_steps:")
        print(t_total)
        print("num_train_epochs:")
        print(num_train_epochs)
        print("\n")
        
        # prepare plm
        no_decay = ['bias', 'LayerNorm.weight']
        plm_parameters = [
            {'params': [p for n, p in self.prompt_model.plm.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.prompt_model.plm.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        # prepare p-tuning parameters
        # we need to remove raw_embedding, otherwise we'll
        # have "same parameter in different groups" error
        # q, what's soft embeddings?
        template_parameters = [{'params': [p for n, p in self.prompt_model.template.named_parameters() if 'raw_embedding' not in n]}]

        plm_optimizer = AdamW(plm_parameters, lr=1e-5, eps=self.args.adam_epsilon)
        plm_scheduler = get_linear_schedule_with_warmup(plm_optimizer, 
            num_warmup_steps=self.args.warmup_steps, 
            num_training_steps=t_total)
        ptuning_optimizer = AdamW(template_parameters, lr=self.args.lr, eps=self.args.adam_epsilon)
        ptuning_scheduler = get_linear_schedule_with_warmup(ptuning_optimizer, 
            num_warmup_steps=self.args.warmup_steps, 
            num_training_steps=t_total)


        best_dev32_acc = 0.0
        best_dev32_one = 0.0
        best_global_step = 0
        best_loss = 0.0
        early_stop_epoch = 0
        tr_loss = 0.0
        global_step = 0

        loss_func = torch.nn.CrossEntropyLoss()
        gradient_acc_steps = self.args.gradient_accumulation_steps
        
        train_iterator = trange(int(num_train_epochs), desc="Epoch")
        for _ in train_iterator:
        # for _ in range(int(num_train_epochs)):

            epoch_iterator = tqdm(self.train_dataloader, desc="Iter")
            for step, batch in enumerate(epoch_iterator):
            # for step, batch in enumerate(self.train_dataloader):
                # print(batch)
                # res = self.model.tokenizer.convert_ids_to_tokens(batch.input_ids[0])
                # print(res)
                # cmp = [2, 59, 46, 35, 14, 157, 16, 14, 991, 9, 3, 14, 157, 16, 14, 4698, 2255, 5657, 102, 9, 14, 1637, 1743, 1637, 60, 4, 3]
                # print(self.tokenizer.convert_ids_to_tokens(cmp))
                # exit(0)
                if self.args.use_cuda:
                    batch = batch.cuda()
                
                logits = self.model.forward(batch)
                labels = batch['label']
                loss = loss_func(logits, labels)

                if gradient_acc_steps > 1:
                    loss = loss / gradient_acc_steps

                loss.backward()
                tr_loss += loss.item()

                if (step + 1) % gradient_acc_steps == 0:

                    torch.nn.utils.clip_grad_norm_(self.prompt_model.parameters(), self.args.max_grad_norm)


                    plm_optimizer.step()
                    plm_scheduler.step()
                    ptuning_optimizer.step()
                    ptuning_scheduler.step()

                    self.prompt_model.zero_grad()
                    global_step += 1

                    if global_step % self.args.eval_every_step == 0:
                        dev32_acc, dev32_one = self.eval(self.valid_dataloader)
                        logger.info(f"current dev32 acc {dev32_acc}, current dev32 one: {dev32_one}")
                        if dev32_acc >= best_dev32_acc:
                            if dev32_acc > best_dev32_acc:
                                early_stop_epoch = 0
                            else:
                                early_stop_epoch += 1

                            best_dev32_acc = dev32_acc
                            best_global_step = global_step
                            best_loss = tr_loss
                            best_dev32_one = dev32_one
                            save_path = os.path.join(self.args.out_dir, self.id + ".ckpt")
                            if self.args.save_model:
                                torch.save(self.prompt_model.state_dict(), save_path)
                            logger.info("best dev32 acc: %.4f | best global step: %d | best global loss %.4f | best dev32 one: %.4f" % \
                                (best_dev32_acc, best_global_step, best_loss/best_global_step, best_dev32_one))
                            
                            # test_acc, test_one = self.eval(self.test_dataloader)
                            # logger.info("eval acc: %.4f" % (test_acc))
                            # logger.info("eval one: %.4f" % (test_one))
                        else:
                            early_stop_epoch += 1

                if global_step > t_total or early_stop_epoch >= 10:
                    logger.info("reach early stop, stopping")
                    epoch_iterator.close()
                    break
            
            if global_step > t_total or early_stop_epoch >= 10:
                logger.info("reach early stop, stopping")
                train_iterator.close()
                break        


if __name__ == '__main__':
    args = parse_args()
    print(args)
    trainer = Trainer(args)
    trainer.train()