# REPOSITORIES

## Additive-Prompt-Tuning

#### Description
This repository implements APT (Attentive Prompt Tuning) combined with the MOMENT foundation model,
based on ["APT: Adaptive Prompt Tuning for Continual Learning" (arxiv:2503.07979)](https://arxiv.org/pdf/2503.07979).  
APT employs an exponential moving average (EMA) strategy to merge prompts
#### Use command example
`python3 run.py --config configs/dailysport.yaml --gpuid 0   --learner_type prompt --learner_name APT_Learner   --prompt_param 0.01 --ema_coeff 0.01 --lr 0.0003   --schedule 10 --seed 3 --overwrite 1`


## CL-Lora-MOMENT

#### Description
This repository implements CL-LoRA (Continual Learning with Low-Rank Adaptation) with the MOMENT foundation model. The approach 
applies LoRA adapters. Each task gets its own set of LoRA adapters while keeping the MOMENT backbone frozen. The 
implementation includes diagonal forward passes for task-specific feature routing and cosine linear classifiers with normalization.

## Coda_Attention

#### Description
This repository implements CODA-Prompt based on   ["CODA-Prompt" (arxiv:2211.13218)](https://arxiv.org/abs/2211.13218)
combined with MOMENT with an attention-based task predictor to address feature
space collapse issues. The approach uses dual prompting (task-specific G-prompts and shared E-prompts).`
#### Use command example
`python3 continual_learning_coda.py --prompt_length 5 --pool_size 20 --top_k 2 --n_tasks 6 --batch_size 16 --epochs_per_task 5`



## L2PROMPT_MOMENT

#### Description
This repository implements continual learning for time series classification using MOMENT 
foundation model with L2-Prompt and multi-head classifiers. The approach uses a frozen MOMENT backbone 
for feature extraction, followed by L2-Prompt for dynamic prompt selection. The architecture includes a
task predictor that automatically identifies which task a sample belongs to during inference, and separate 
classification heads per task to prevent catastrophic forgetting. The L2-Prompt pool selects relevant 
prompts based on cosine similarity between input queries and learned keys, enabling knowledge sharing 
across tasks while maintaining task-specific adaptations.
#### Use command example
`python3 continual_learning.py --prompt_length 5 --pool_size 20 --top_k 2 --n_tasks 6 --batch_size 16 --epochs_per_task 5`


## LETS-C-Approach


#### Description
This repository combines CODA-Prompt with the LETS methodology from 
["Language-Enhanced Time Series Classification" (arxiv:2407.06533)](https://arxiv.org/abs/2407.06533). 
without Moment. The TextLETS encoder converts time series to text using LLaMA, which is then processed through the prompting mechanism.
The architecture includes a task predictor.
#### Use command example
`python3 continual_learning.py --prompt_length 5 --pool_size 20 --top_k 2 --n_tasks 6 --batch_size 16 --epochs_per_task 5`


## LETS-C-Approach-Plus-MOMENT

#### Description
This repository combines the MOMENT 
foundation model with L2-Prompt and integrates the LETS methodology 
from ["Language-Enhanced Time Series Classification" (arxiv:2407.06533)](https://arxiv.org/abs/2407.06533),
which converts time series into text representations using LLaMA encoders. 
The basic approach (`main.py`) uses a frozen MOMENT backbone with a learnable prompt pool 
that dynamically selects prompts based on input similarity. The TextLETS approach (`textlets_multihead.py`)
combines text-based embeddings with MOMENT, using multi-head classifiers and task prediction.
#### Use command example
`python3 continual_learning.py --prompt_length 5 --pool_size 20 --top_k 2 --n_tasks 6 --batch_size 16 --epochs_per_task 5`




## Dataset Access

The DailySport datasets used in this project are available here:

**[Download Datasets (Google Drive)](https://drive.google.com/drive/folders/17UOvzTffzVzzLT3EIegKJ8ug-VvnsQ4Z?usp=drive_link)**  
