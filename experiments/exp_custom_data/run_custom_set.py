import os
import argparse
import json

import torch
from tqdm import tqdm
from openprompt.data_utils.utils import InputExample

from models import PtuningArgs, PtuningModel





def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt", type=str, 
        default="tmp/models_ensemble-train64/2022-07-18-00-06_wic_template16_albert-xxlarge-v2.ckpt", 
        required=False)
    parser.add_argument("--config", type=str, 
        default="tmp/config_ensemble-train64/wic_template16_albert-xxlarge-v2.cfg", 
        required=False)
    parser.add_argument("--write_to_file", type=str, default="T", choices=["T", "F"])
    parser.add_argument("--out_dir", type=str, default="tmp/analysis_custom_data/")
    parser.add_argument("--test_filepath", type=str, default="tmp/custom_data/custom_test.jsonl")

    args = parser.parse_args()
    return args



class CustomDataRunner:

    def __init__(self, args: argparse.ArgumentParser):
        self.model = PtuningModel(PtuningArgs.fromFile(args.config))
        self.model.prompt_model.load_state_dict(torch.load(args.ckpt))
        self.ckpt = os.path.basename(args.ckpt)
        self.out_dir = args.out_dir
        self.test_filepath = args.test_filepath
        self.stop_idx = 0

        correct_preds = 0
        total_preds = 0
        verbose_output = []

        with open(self.test_filepath, "r") as data_f:
            data = [json.loads(line) for line in data_f] #list of dicts

            for instance in data:
                s1 = str(instance['sentence1']).strip()
                s2 = str(instance['sentence2']).strip()
                word = str(instance['word']).strip()
                label_str = str(instance['label']).strip()
                label_int = [1 if label_str=="True" else 0][0]
                idx = instance['idx']
                try:
                    instance_type = instance['type']
                except:
                    instance_type = ""

                op_input = self.load_one_instance(s1, s2, word, label_int, idx)
                if self.model.args.use_cuda:
                    op_input = op_input.cuda()

                logits = self.model.forward(op_input)
                pred_int = torch.argmax(logits, dim=-1).cpu().tolist()[0]
                pred_str = ["True" if pred_int==1 else "False"][0]

                total_preds += 1
                if pred_int==label_int:
                    pred_is = "Correct"
                    correct_preds += 1
                else:
                    pred_is = "Wrong"

                str_to_output = f"{pred_is} - {label_str} - {pred_str} - {instance_type} - {s1} - {s2} - {word} - {idx}"

                verbose_output.append(str_to_output)

                #self.stop_idx+=1 #tmp
                #if self.stop_idx == 10: #tmp
                    #break #tmp
            
        acc = correct_preds/total_preds
        

        if args.write_to_file == "T":
            from datetime import datetime
            ts = datetime.today().strftime('%Y-%m-%d-%H-%M')
            out_path = os.path.join(self.out_dir, f"ct-preds_{ts}.txt")
            with open(out_path, "w+") as f:
                for output in verbose_output:
                    f.write(f"{output}\n")
                
                f.write(f"\n\nAccuracy: {acc}")
                


    def load_one_instance(self, s1, s2, word, label, idx):
        # transform ONE natural language sentence into OpenPrompt input

        # ct: custom test
        sample = [InputExample(guid=f"ct-{idx}", 
            text_a=s1.strip(),
            text_b=s2.strip(),
            meta={"word": word.strip()},
            label=label)]

        data_loader = self.model.build_single_dataloader(
            dataset=sample,
            batch_size=1,
        )

        return next(iter(data_loader))




if __name__ == '__main__':
    args = parse_args()
    analyzer = CustomDataRunner(args)