# LLM Pretrain
***Duration***: 2025 Nov 07 - 2025 Dec 14  
***Curriculum***: Training - Inference  
***Imitated LLM Model***: `GPT 2`  


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
Definition
(1) Vocab (2) Context Length (3) Epoch (4) Batch Size (5) FFNN Hidden Dim (6) Num of layers (7) Num of Attention Head

### b. Embedding
(1) Token Embedding  

**Token Vocab & Tokenizer**: `Tiktokenizer`  
**Dataset**: Pre-Processed `Harry Potter from Kaggle`  
**Logic**: Token Embedding `Dataset` by `Tokenizer`  

(2) Position Embedding  

Context Length x EMB_DIM

### c. Dropout
(1) Making Noise  
**Built-In Function of PyTorch**: `nn.dropout`

### d. Transformer Block
> 1. **Multi Head Attention**  
> (1.a) Layer Normalization  
> (1.b) Attention_Score  
> (1.c) Dropout  
> (1.d) Residual  
> 
> 2.  **FFNN**  
> (2.a) Layer Normalization  
> (2.b) FFNN  
> (2.c) Dropout  
> (2.d) Residual   

#### Shared Components
*1. LayerNormalization*

Purpose: for Sustainability  
Logic: Shift and Scale  

*2. Dropout*

Purpose: for Preventing Overfitting  
Logic: Built-in Function (Modify Drop_Rate)  

*3. Residual*

Purpose: for Including previous activation 
Logic: a = a + b  

#### Separate Components
*1. Attention_Score (Multi-Head Attention)*

Purpose: for Computing pairwise relevance between tokens | mixing across features  
Logic: Attention(Q,K,V) = softmax(QxK⊤)*V  

*2. FFNN*

Purpose: non-linearization between each tokens  
Logic: GeLU | ReLU

### e. Final Normalization



### f. logits 



---
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
