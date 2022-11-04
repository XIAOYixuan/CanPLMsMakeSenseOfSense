import os
import argparse
from turtle import position

import numpy as np

import torch
from tqdm import tqdm
from openprompt.data_utils.utils import InputExample

from sklearn.metrics.pairwise import cosine_similarity

from models import PtuningArgs, PtuningModel

from wic_util.data_loader import WicFewShotLoader

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="outputs/")

    args = parser.parse_args()
    return args


class RepExtractor:

    def __init__(self, args: argparse.ArgumentParser):
        args = PtuningArgs.fromFile(args.config)
        self.model = PtuningModel(args)

        wic_data = WicFewShotLoader(args.data_dir)
        self.guid2text = {}
        self._build_guid_map(wic_data.train)
        self._build_guid_map(wic_data.valid)
        self._build_guid_map(wic_data.test)
        # print(self.guid2text["dev-0"])


    def _build_guid_map(self, samples):
        for sample in samples:
            guid = sample.guid
            guid = guid.replace("val-", "dev-")
            self.guid2text[guid] = sample


    def match_tokens(self, input_ids, full_input_ids, find_sep=False):
        input_ids = input_ids[1:-1]
        id_len = len(input_ids)
        st, ed = None, None
        
        total_len = len(full_input_ids)
        pt = 0
        if find_sep:
            while pt < total_len:
                # print(full_input_ids[pt])
                if full_input_ids[pt] == 3:
                    break
                pt += 1
        while pt < total_len:
            if full_input_ids[pt:pt+id_len].tolist() == input_ids:
                st, ed = pt, pt+id_len
                break
            pt += 1
        assert st is not None
        return (st, ed)


    def debug(self, word_a_pos, word_b_pos, inputs):
        st, ed = word_a_pos
        word_a = inputs[st:ed]
        st, ed = word_b_pos
        word_b = inputs[st:ed]
        tokens = self.model.tokenizer.convert_ids_to_tokens(word_a)
        print("--- word a", word_a_pos, tokens)
        tokens = self.model.tokenizer.convert_ids_to_tokens(word_b)
        print("--- word b", word_b_pos, tokens)
        exit(0)

    

    def extract_target_word_ids(self, tokens_batch):
        positions = []
        input_ids = tokens_batch["input_ids"]
        guids = tokens_batch["guid"]
        for guid, inputs in zip(guids, input_ids):
            assert guid in self.guid2text
            wic_sample = self.guid2text[guid]
            word_a = self.model.tokenizer.encode(wic_sample.word_a)
            word_a_pos = self.match_tokens(word_a, inputs)
            word_b = self.model.tokenizer.encode(wic_sample.word_b)
            word_b_pos = self.match_tokens(word_b, inputs, find_sep=True)
            positions.append([word_a_pos, word_b_pos])
            # self.debug(word_a_pos, word_b_pos, inputs)
        return positions


    def extract_embedding_single(self, position, embeddings):
        def extract_seg(pos):
            st, ed = pos
            return embeddings[st: ed]
        text_a = extract_seg(position[0]) 
        text_a = text_a.detach().numpy().sum(axis=0)
        text_b = extract_seg(position[1]) 
        text_b = text_b.detach().numpy().sum(axis=0)
        return text_a, text_b
                

    def extract_embedding(self, positions, embeddings):
        # [batch_size, 3]
        assert(len(positions) == embeddings.size(0))
        embed_cpu = embeddings.cpu()
        word_as, word_bs = [], []

        for bid in range(len(positions)):
            if positions[bid] is None:
                continue
            word_a, word_b = self.extract_embedding_single(positions[bid], embed_cpu[bid])
            # word_reps.append([word_a, word_b])
            word_as.append(word_a)
            word_bs.append(word_b)

        return (np.asarray(word_as), np.asarray(word_bs))


    def filter_by_surface(self, guids, embeds, hiddns, labels):
        nembeds = []
        nhiddns = []
        nlabels = []

        for pt, guid in enumerate(guids):
            wic_sample = self.guid2text[guid]
            tgt, wa, wb = wic_sample.tgt_word, wic_sample.word_a, wic_sample.word_b
            tgt, wa, wb = tgt.lower(), wa.lower(), wb.lower()
            if tgt != wa or tgt != wb: 
                continue
            nembeds.append(embeds[pt])
            nhiddns.append(hiddns[pt])
            nlabels.append(labels[pt])
        if len(nembeds) == 0:
            return None, None, None
        return np.vstack(nembeds), np.vstack(nhiddns), np.vstack(nlabels)


    def process_one_batch(self, sample):
        positions = self.extract_target_word_ids(sample) 
        guids = sample["guid"]
        sample = sample.cuda()
        self.model.forward(sample)

        hidden_states = self.model.my_verbalizer.hidden_states
        embed_reps = self.extract_embedding(positions, hidden_states[0])
        hiddn_reps = self.extract_embedding(positions, hidden_states[-1])
        
        embed_score = cosine_similarity(embed_reps[0], embed_reps[1])
        hiddn_score = cosine_similarity(hiddn_reps[0], hiddn_reps[1])
        labels = sample['label'].cpu().detach().numpy().reshape((-1, 1))
        
        # print('embed size', len(embed_reps), embed_reps[0].shape, \
        #     'labels', labels.shape)
        embed_score, hiddn_score, labels = self.filter_by_surface(
            guids,
            embed_score, hiddn_score, labels)
        return embed_score, hiddn_score, labels   


    def batch_forward(self, data_loader, tag="vector_cmp/full_zeroshot/test"):
        embed_scores = []
        hiddn_scores = []
        labels = []
        for sample in tqdm(data_loader):
            # print(f"--------------------- {sample['guid']}")
            embed, hiddn, label = self.process_one_batch(sample)
            if embed is None:
                continue
            embed_scores.append(embed)
            hiddn_scores.append(hiddn)
            labels.append(label)
        embed_scores = np.vstack(embed_scores)
        hiddn_scores = np.vstack(hiddn_scores)
        labels = np.vstack(labels)
        print(embed_scores.shape, hiddn_scores.shape, labels.shape)
        print(f"saving to {tag}...")
        np.save(tag+"_embed", embed_scores)
        np.save(tag+"_hiddn", hiddn_scores)
        np.save(tag+"_label", labels)
        

if __name__ == '__main__':
    args = parse_args()
    rep_extractor = RepExtractor(args)
    dir = "vector_cmp/same_train/"
    rep_extractor.batch_forward(rep_extractor.model.valid_dataloader, 
        tag=f"{dir}/valid")
    rep_extractor.batch_forward(rep_extractor.model.test_dataloader,
        tag=f"{dir}/test")