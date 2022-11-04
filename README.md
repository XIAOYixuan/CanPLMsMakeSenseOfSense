# Can PLMs Make Sense of Sense? 

## Advanced Machine Learning for Speech and Language Processing [2022] - project


### Abstract

Prompt-based Learning is promising because it provides an avenue to solve NLP tasks efficiently. Tasks can be phrased as fill-in-the-gap strings and pre-trained models can be prompted directly.

We study the WiC task, which involves disambiguating words in context, and has posed issues to previous works. 

We aim at understanding how the task should be presented to models; we do so by experimenting with different prompts. 

We find that significantly improved results can be achieved by correctly separating the prompt components, using two soft tokens (for two context sentences), and not indicating the target word explicitly.


### Data

We use the FewGlue WiC dataset (https://github.com/THUDM/P-tuning/tree/main/FewGLUE_32dev/WiC), which is designed for Few-shot learning, to align with Liu et al. (2021) and Schick & Schütze (2020). 

Both the training set and the validation set have 32 samples each and are subsets of the original SuperGLUE \cite{superglue} training set. The test set is the SuperGLUE validation set, which has 638 samples. 

All partitions are balanced across the two classes (true/false).


### Table of Contents

- external_src: external resources that we built upon

- src: our own source code

- data

  - FewGlue32dev

- explore: miscellaneous

- experiments:

  - exp_demonstrations: we added demonstrations (one per class i.e., one 'true' and 'false') to each test sample. There are two settings for this experiment:
    
    1) fixed demonstations: the <ins>same two</ins> demonstrations (randomly picked) are attached to all test samples.
    
    2) we try to find the two best demonstrations for each test sample and attached them.
    
  
  - exp_custom_data: we designed a dataset that has the same format as WiC, but is made up of 'tricky' examples, as well as 'normal' instances (examples in the respective README).
  
  - exp_ensemble: we ensembled different models and different templates and aggregated the predictions to leverage the sense knowledge of different sources.



### Requirements

OpenPrompt, PET, ...


### How to Run
...

