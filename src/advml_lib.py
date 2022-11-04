from transformers import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.file_utils import ModelOutput
from yacs.config import CfgNode
from openprompt.data_utils import InputFeatures
from openprompt import Verbalizer
from typing import *
import torch
import torch.nn.functional as F
from openprompt.utils.logging import logger
from openprompt.prompts import ManualVerbalizer, ManualTemplate, PtuningTemplate


class WiCPtuningTemplate(PtuningTemplate):

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, text: Optional[List[str]] = None, prompt_encoder_type: str = "lstm", placeholder_mapping: dict = ...):
         super().__init__(model, tokenizer, text, prompt_encoder_type, placeholder_mapping)
         
         
    def on_text_set(self):
        """ When template text was set, parse the text, and generate parameter needed
        """



class SynsetVerbalizer(ManualVerbalizer):

    def __init__(self,
                tokenizer: PreTrainedTokenizer,
                classes: Optional[List] = None,
                num_classes: Optional[Sequence[str]] = None,
                label_words: Optional[Union[Sequence[str], Mapping[str, str]]] = None,
                prefix: Optional[str] = " ",
                multi_token_handler: Optional[str] = "first",
                post_log_softmax: Optional[bool] = True,
            ):
            super().__init__(tokenizer, classes, num_classes, label_words, prefix, multi_token_handler, post_log_softmax)
            self.vocab_logits = None
            self.topk = None


    def gather_outputs(self, outputs: ModelOutput):
        r""" retrieve useful output for the verbalizer from the whole model ouput
        By default, it will only retrieve the logits

        Args:
            outputs (:obj:`ModelOutput`) The output from the pretrained language model.

        Return:
            :obj:`torch.Tensor` The gathered output, should be of shape (``batch_size``,
            ``seq_len``, ``any``)
        """
        self.hidden_states = outputs.hidden_states
        return outputs.logits


    def process_logits(self, logits: torch.Tensor, **kwargs):
        self.vocab_logits = logits
        _, self.topk = logits.topk(5, largest=True)
        return super().process_logits(logits, **kwargs)