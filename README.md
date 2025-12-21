# LLM Pretrain
Imitated LLM Model: `GPT 2`  
Curriculum: Training - Inference

## Training Summarize
> 1. PROCESSING Datasets 
> 2. MAIN Logic
> 3. Training Data

## 1. PREPARE Datasets
Source of Dataset: Harry Potter from Kaggle  

### LIBRARY
a. re  
b. clean-text  
c. panda  

### a. Clean-text
No function, remove the line-break.

### b. re
Cleaning Data Text

## 2. Main Logic
### List
a. Preparation
b. Embedding
c. Dropout
d. Transformer Block
e. Finalization
f. logits

### a. Preparation
Define   
(1) Vocab (2) Context Length (3) Epoch (4) Batch Size (5) FFNN Hidden Dim (6) Num of layers (7) Num of Attention Head

### b. Embedding
(1) Token Embedding  

**Token Vocab & Tokenizer**: `Tiktokenizer`  
**Dataset**: Pre-Processed `Harry Potter from Kaggle`  
**Logic**: Token Embedding `Dataset` by `Tokenizer`  

(2) Position Embedding
Context Length x EMB_DIM

### Tokenize & DataLoader
### TikTokenizer
good for eng

### DataLoader
#### Input & Target
Input에 따른 Target 값을 추측하게 만든다.

## AFTER Process
### Losses Graph
![훈련 손실 그래프](images/loss.png)




## Inference

--- 
Github: 

## Sources
### Main
https://github.com/HongLabInc/HongLabLLM/blob/main/01_pretraining.ipynb
https://www.youtube.com/watch?v=osv2csoHVAo&t=724s

### Dataset Splitting
https://pozalabs.github.io/Dataset_Splitting/
