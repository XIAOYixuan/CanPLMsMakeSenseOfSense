import os
from os.path import join, abspath, dirname
from pickletools import optimize
from re import template
from turtle import forward
from unicodedata import decimal
import torch

from openprompt import PromptForClassification, PromptDataLoader, PromptForGeneration
from openprompt.prompts import ManualVerbalizer, PtuningTemplate, ManualTemplate, GenerationVerbalizer
from openprompt.data_utils.utils import InputExample
from openprompt.plms import load_plm

from advml_lib import SynsetVerbalizer

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

from tqdm import trange, tqdm

import log


class WiCModelArgs:

    def __init__(self, 
            task_name, data_dir,
            template, labels, 
            model_name, model_path,
            ckpt, 
            max_seq_len,
            train_batch_size,
            eval_batch_size,
            use_cuda):
        self.task_name = task_name
        self.data_dir = data_dir

        self.template = template
        self.labels = labels 

        self.model_name = model_name
        self.model_path = model_path
        if ckpt == "none":
            self.ckpt = None
        else:
            self.ckpt = ckpt

        self.max_seq_len = max_seq_len
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size

        self.use_cuda = True

        self.load_data = True

    
    def __str__(self):
        def make_str(key, value):
            return f"{key}={value}\n"
        
        return make_str("task_name", self.task_name) + \
            make_str("data_dir", self.data_dir) + \
            make_str("template", self.template) + \
            make_str("labels", self.labels) + \
            make_str("model_name", self.model_name) + \
            make_str("model_path", self.model_path) + \
            make_str("ckpt", self.ckpt) + \
            make_str("max_seq_len", self.max_seq_len) + \
            make_str("train_bz", self.train_batch_size) + \
            make_str("eval_bs", self.eval_batch_size)


    @classmethod
    def fromFile(cls, file_path):
        cfg_dict = {}
        with open(file_path) as fp:
            for line in fp:
                key, val = line.strip().split()
                cfg_dict[key] = val
        return cls(cfg_dict["task_name"], cfg_dict["data_dir"], 
                cfg_dict["template"], cfg_dict["labels"], 
                cfg_dict["model_name"], cfg_dict["model_path"],
                cfg_dict["ckpt"],
                int(cfg_dict["max_seq_len"]), 
                int(cfg_dict["train_batch_size"]), 
                int(cfg_dict["eval_batch_size"]), 
                cfg_dict["use_cuda"] == "True")
        


class WiCBaseModel:

    def __init__(self, args: WiCModelArgs):
        self.my_template = None
        self.my_verbalizer = None
        self.load_data()
        self.prepare_template()
        self.prepare_verbalizer()
        self.build_dataloader()
        self.build_prompt_model()
        if args.ckpt is not None:
            self.prompt_model.load_state_dict(torch.load(args.ckpt))
        print("ckpt", args.ckpt)
        # self.show_parameters()


    def extract_sentence(self, sample):
        input_ids = sample["input_ids"]
        print('--- label', sample["label"])
        clean_ids = []
        clean_tokens = []
        for tokens in input_ids:
            tokens = tokens.cpu().tolist()

            clean = [x for x in tokens if x != 0]
            clean_ids.append(clean)

            tokens = self.tokenizer.convert_ids_to_tokens(clean)
            clean_tokens.append(tokens)

            sent = self.tokenizer.decode(clean)
            print('----', sent)
        return clean_ids, clean_tokens

        
    def load_data(self):
        raise NotImplementedError("base class: load data")


    def forward(self, inputs):
        raise NotImplementedError("base class: fwd")


    def plm(self):
        return self.prompt_model.plm

    def template(self):
        return self.prompt_model.template()


    def prepare_template(self):
        raise NotImplementedError("base class: unk template")


    def prepare_verbalizer(self):
        raise NotImplementedError("base class: unk verbalizer")


    def build_prompt_model(self):
        raise NotImplementedError("base class: unk dataloader")

    
    def build_dataloader(self):
        raise NotImplementedError("base class: unk dataloader")


    def show_parameters(self):
        print("prompt model------------------------------")
        for n, p in self.prompt_model.template.named_parameters():
            print(n, p.size())
        print("plm------------------------------")
        for n, p in self.prompt_model.plm.named_parameters():
            print(n, p.size())


    def build_single_dataloader(self, 
            dataset, 
            batch_size, 
            verbalizer = None, 
            shuffle=False, 
            decoder_max_l = -1, 
            teacher_forcing=False, 
            predict_eos=False):
        return PromptDataLoader(
            dataset=dataset,
            template=self.my_template,
            verbalizer=verbalizer,
            tokenizer=self.tokenizer,
            tokenizer_wrapper_class=self.WrapperClass,
            max_seq_length=self.args.max_seq_len,
            decoder_max_length=decoder_max_l,
            batch_size=batch_size,
            shuffle=shuffle,
            teacher_forcing=teacher_forcing,
            predict_eos_token=predict_eos
        )


class PtuningArgs(WiCModelArgs):

    def __init__(self, task_name, data_dir, template, labels, model_name, model_path, ckpt, max_seq_len, train_batch_size, eval_batch_size, use_cuda):
        super().__init__(task_name, data_dir, template, labels, model_name, model_path, ckpt, max_seq_len, train_batch_size, eval_batch_size, use_cuda)


class PtuningModel(WiCBaseModel):

    def __init__(self, args: PtuningArgs):
        self.args = args
        super().__init__(args)


    def load_data(self):
        self.label2int = {}
        self.label_words = []

        with open(self.args.labels) as fp:
            val = 0
            for line in  fp:
                key, verbalizers  = line.strip().split(':')
                self.label2int[key.strip()] = val
                val += 1 
                verbalizers = verbalizers.strip().split(',')
                verbalizers = [x.strip() for x in verbalizers]
                self.label_words.append(verbalizers)
        print('label2int', self.label2int)
        print('label words', self.label_words)

        from data_wrapper import DataWrapper
        data_loader = DataWrapper(self.args)

        self.datasets = {}
        for split in ["train", "valid", "test"]:
            samples = []
            for data in data_loader.dataset[split]:
                int_label = self.label2int[data.label]
                input_example = InputExample( guid=data.guid,
                    text_a=data.text_a, \
                    text_b=data.text_b, \
                    meta=data.meta , \
                    label=int_label)
                # print(input_example)
                samples.append(input_example)
            self.datasets[split] = samples
            print('done with', split)


    def forward(self, inputs):
        return self.prompt_model(inputs)


    def prepare_template(self):
        # prepare template

        # TODO: simplify this, use just model_path
        self.plm, self.tokenizer, self.model_cfg, self.WrapperClass = load_plm(self.args.model_name, self.args.model_path)

        self.my_template = PtuningTemplate(
            model=self.plm,
            tokenizer=self.tokenizer,
            prompt_encoder_type='lstm'
        ).from_file(self.args.template)
        print("-----template-----")
        print(self.args.template)
    
    
    def prepare_verbalizer(self):
        # self.my_verbalizer = ManualVerbalizer(
        self.my_verbalizer = SynsetVerbalizer(
            self.tokenizer,
            num_classes=len(self.label_words),
            label_words=self.label_words
        )


    def build_prompt_model(self):
        import time
        self.prompt_model = PromptForClassification(
            plm=self.plm,
            template=self.my_template,
            verbalizer=self.my_verbalizer,
            freeze_plm=False
        )

        # TODO: use to device
        fre, ocp = torch.cuda.mem_get_info(device=0)
        we_need = 8000
        cnt = 0
        while fre < 1024*1024*we_need:
            fre, ocp = torch.cuda.mem_get_info(device=0)
            time.sleep(1)
            fremb = fre/1024/1024
            cnt += 1
            if cnt % 60 == 0:
                print(f"waiting for {cnt}s free mem {fremb}")
            continue

        if self.args.use_cuda:
            self.prompt_model = self.prompt_model.cuda()

    
    def build_dataloader(self):
        self.train_dataloader = self.build_single_dataloader(
            dataset=self.datasets["train"],
            batch_size=self.args.train_batch_size,
            shuffle=True
        )

        self.valid_dataloader = self.build_single_dataloader(
            dataset=self.datasets["valid"],
            batch_size=self.args.eval_batch_size,
            shuffle=False
        )

        self.test_dataloader = self.build_single_dataloader(
            dataset=self.datasets["test"],
            batch_size=self.args.eval_batch_size,
            shuffle=False
        )



class DirectCmpModel(PtuningModel):
    """ For template, {"meta": "worda"} and {"meta": "wordb} {soft}
    """

    def __init__(self, args: PtuningArgs):
        super().__init__(args)


    def load_data(self):
        self.label_words = []
        self.guid2inputsample = {}

        with open(self.args.labels) as fp:
            val = 0
            for line in  fp:
                key, verbalizers  = line.strip().split(':')
                val += 1 
                verbalizers = verbalizers.strip().split(',')
                verbalizers = [x.strip() for x in verbalizers]
                self.label_words.append(verbalizers)
        print('label words', self.label_words)

        from wic_util.data_loader import WicFewShotLoader
        wic_fewshot = WicFewShotLoader(self.args.data_dir)
        datasets = {}
        datasets["train"] = wic_fewshot.train
        datasets["valid"] = wic_fewshot.valid
        datasets["test"] = wic_fewshot.test

        self.datasets = {}
        for split in ["train", "valid", "test"]:
            samples = []
            for data in datasets[split]:
                self.guid2inputsample[data.guid] = data
                meta = {
                    "word": data.tgt_word,
                    "word_a": data.word_a,
                    "word_b": data.word_b
                }
                input_example = InputExample( guid=data.guid,
                    text_a=data.text_a, \
                    text_b=data.text_b, \
                    meta=meta , \
                    label=data.label)
                # print(input_example)
                samples.append(input_example)
            self.datasets[split] = samples
            print('done with', split)


class PositionAwareDataLoader(PromptDataLoader):

    def __init__(self, dataset, template, tokenizer_wrapper=None, \
        tokenizer=None, tokenizer_wrapper_class=None, \
        verbalizer=None, max_seq_length=512, batch_size=1, \
        shuffle=False, teacher_forcing=False, decoder_max_length= -1, \
        predict_eos_token=False, truncate_method="tail", \
        drop_last=False, guidmap=None):
        self.guid2inputsample = guidmap
        self.tokenizer = tokenizer
        super().__init__(dataset, template, tokenizer_wrapper, tokenizer, tokenizer_wrapper_class, verbalizer, max_seq_length, batch_size, shuffle, teacher_forcing, decoder_max_length, predict_eos_token, truncate_method, drop_last)

    def find_position(self, full, prefix, tgt, sample=None):
        pts = []
        pt = 0
        # print(prefix, full)
        while pt < len(prefix):
            if prefix[pt] == full[pt]:
                pt += 1
                # print("equal")
                continue
            break
        # print(pt, full[pt:])
        tgt_pt = 0
        while tgt_pt < len(tgt):
            if tgt[tgt_pt] == full[pt+tgt_pt]:
                pts.append(pt+tgt_pt)
                tgt_pt += 1
                continue
            break
        # print(tgt, pts)
        if len(pts) != len(tgt):
            print("found one")
            pts = [pt+i for i in range(len(tgt))]
            # print(full)
            # print(prefix)
            # print(tgt)
            # print(sample)
            #  print(tokenizer.decode([23213, 8593]))
            # print(tokenizer.decode([13265, 242, 857]))

        # assert len(pts) == len(tgt)

        # add one to pts cuz the leading cls token occupies a position
        add_one = [p + 1 for p in pts]
        return add_one


    def replace_position_ids(self, input_ids, position_ids, vis, tgt_ids, tgt_pos):
        len_tgt = len(tgt_ids)
        # print(tgt_ids, input_ids)
        assert len(tgt_ids) == len_tgt

        def fill_vec(vec, st, n, val):
            for i in range(n):
                vec[i+st] = val

        for i in range(len(input_ids), -1, -1):
            if input_ids[i:i+len(tgt_ids)] == tgt_ids and vis[i] == False:
                fill_vec(vis, i, len_tgt, True)
                # print("found it")
                position_ids[i:i+len_tgt] = tgt_pos[0:len_tgt]
                break


    def _add_position_ids(self, inputexample, input_ids):
        a_st = inputexample.word_a_char_pos[0]
        b_st = inputexample.word_b_char_pos[0]
        # TODO: hard code
        prefix_a = inputexample.text_a[:a_st]
        prefix_a = prefix_a.lower()
        prefix_b = inputexample.text_a + " / " + inputexample.text_b[:b_st]
        prefix_b = prefix_b.lower()
        
        a_input_ids = self.tokenizer.encode(prefix_a)[1:-1]
        b_input_ids = self.tokenizer.encode(prefix_b)[1:-1]
        #print(a_input_ids)
        #print(b_input_ids)
        word_a = self.tokenizer.encode(inputexample.word_a.lower())[1:-1]
        word_b = self.tokenizer.encode(inputexample.word_b.lower())[1:-1]
        debug_dict = {
            "word_a": word_a, 
            "word_b": word_b,
            "prefix_b": prefix_b,
            "pos_b": b_st,
            "just_b": inputexample.text_b[b_st:]
        }
        pos_a = self.find_position(input_ids[1:-1], a_input_ids, word_a, debug_dict)
        pos_b = self.find_position(input_ids[1:-1], b_input_ids, word_b, debug_dict)
        position_ids = list(range(len(input_ids)))
        vis = [False for i in range(len(input_ids))]
        self.replace_position_ids(input_ids, position_ids, vis, word_b, pos_b)
        self.replace_position_ids(input_ids, position_ids, vis, word_a, pos_a) 
        assert len(input_ids) 
        return position_ids


    def tokenize(self) -> None:
        from openprompt.data_utils import InputFeatures
        r"""Pass the wraped text into a prompt-specialized tokenizer,
           the true PretrainedTokenizer inside the tokenizer is flexible, e.g. AlBert, Bert, T5,...
        """
        for idx, wrapped_example in tqdm(enumerate(self.wrapped_dataset),desc='tokenizing'):
        # for idx, wrapped_example in enumerate(self.wrapped_dataset):
            inputfeatures = InputFeatures(**self.tokenizer_wrapper.tokenize_one_example(wrapped_example, self.teacher_forcing), **wrapped_example[1]).to_tensor()
            sample = inputfeatures
            input_ids = sample["input_ids"].tolist()
            guid = str(sample["guid"])
            # print(input_ids)
            input_example = self.guid2inputsample[guid]
            position_ids = self._add_position_ids(input_example, input_ids)
            position_ids = torch.tensor(position_ids)
            sample["position_ids"] = position_ids
            # print(input_ids, position_ids)
            self.tensor_dataset.append(sample)


class PositionAwareModel(DirectCmpModel):

    def __init__(self, args: PtuningArgs):
        self.args = args
        # self.my_template = None
        # self.my_verbalizer = None
        # self.load_data()
        # self.prepare_template()
        # self.prepare_verbalizer()
        # self.build_dataloader()
        super().__init__(args)

    
    def build_single_dataloader(self, 
            dataset, 
            batch_size, 
            verbalizer = None, 
            shuffle=False, 
            decoder_max_l = -1, 
            teacher_forcing=False, 
            predict_eos=False):
        return PositionAwareDataLoader(
            dataset=dataset,
            template=self.my_template,
            verbalizer=verbalizer,
            tokenizer=self.tokenizer,
            tokenizer_wrapper_class=self.WrapperClass,
            max_seq_length=self.args.max_seq_len,
            decoder_max_length=decoder_max_l,
            batch_size=batch_size,
            shuffle=shuffle,
            teacher_forcing=teacher_forcing,
            predict_eos_token=predict_eos,
            guidmap=self.guid2inputsample
        )



class GenArgs(WiCModelArgs):
    
    def __init__(self, task_name, data_dir, template, labels, model_name, model_path, max_seq_len, train_batch_size, eval_batch_size, use_cuda):
        super().__init__(task_name, data_dir, template, labels, model_name, model_path, max_seq_len, train_batch_size, eval_batch_size, use_cuda)
        # TODO: extends to gpt
        self.model_name = "t5-lm"
        self.model_path = "google/t5-base-lm-adapt"
        self.tune_plm = False
        self.train_max_decoder_length = 10
        self.eval_max_decoder_length = 3
        self.generation_arg = {
            "max_length": self.train_max_decoder_length
        }


class GenModel(WiCBaseModel):
    """ WiC with a generative model
    """

    def __init__(self, args: GenArgs):
        self.args = args
        super().__init__(args)


    def load_data(self):
        from data_wrapper import DataWrapper
        data_loader = DataWrapper(self.args)
        # TODO: read it from config file
        self.label2int = {
            "F": 0,
            "T": 1
        }

        self.datasets = {}
        for split in ["train", "valid", "test"]:
            samples = []
            for data in data_loader.dataset[split]:
                int_label = self.label2int[data.label]
                input_example = InputExample( guid=data.guid,
                    text_a=data.text_a, \
                    text_b=data.text_b, \
                    meta=data.meta , \
                    label=int_label)
                # print(input_example)
                samples.append(input_example)
            self.datasets[split] = samples
            print('done with', split)


    def prepare_template(self):
        self.plm, self.tokenizer, self.model_cfg, \
            self.WrapperClass = load_plm(self.args.model_name, \
                self.args.model_path)

        # TODO: ptuning
        self.my_template = ManualTemplate(
            tokenizer=self.tokenizer
        ).from_file(self.args.template)


    def prepare_verbalizer(self):
        # InputSample, 1 for yes, 0 for no
        # TODO: don't hard code this
        class_labels = ['Same', 'Different']
        self.my_verbalizer = GenerationVerbalizer(
            tokenizer=self.tokenizer,
            classes=class_labels,
            is_rule=True
        ).from_file("templates/wic.gen.labels.txt", choice=0)


    def build_prompt_model(self):
        self.prompt_model = PromptForGeneration(
            plm=self.plm, 
            template=self.my_template,
            freeze_plm=not self.args.tune_plm,
            plm_eval_mode=True
        )

        if self.args.use_cuda:
            self.prompt_model = self.prompt_model.cuda()


    def build_dataloader(self):
        self.train_dataloader = self.build_single_dataloader(
            verbalizer=self.my_verbalizer,
            dataset=self.datasets["train"],
            batch_size=self.args.train_batch_size,
            shuffle=True,
            teacher_forcing=True,
            predict_eos=True,
            decoder_max_l=self.args.train_max_decoder_length
        )

        self.valid_dataloader = self.build_single_dataloader(
            verbalizer=self.my_verbalizer,
            dataset=self.datasets["valid"],
            batch_size=self.args.eval_batch_size,
            shuffle=False,
            teacher_forcing=False,
            predict_eos=False,
            decoder_max_l=self.args.eval_max_decoder_length
        )

        self.test_dataloader = self.build_single_dataloader(
            verbalizer=self.my_verbalizer,
            dataset=self.datasets["test"],
            batch_size=self.args.eval_batch_size,
            shuffle=False,
            teacher_forcing=False,
            predict_eos=False,
            decoder_max_l=self.args.eval_max_decoder_length
        )


    def forward(self, inputs):
        raw_seq, sentences = self.prompt_model.generate(
            inputs,
            **self.args.generation_arg
        )
        return raw_seq, sentences