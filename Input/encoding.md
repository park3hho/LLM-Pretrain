## GPU Acceleration
```
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```
### GPU Optimization
1. Kernel Optimization
2. Operation Optimization
3. Model Compression
4. CUDA Tunning and Optimization

### 1. Kernel Optimization
CUDA Kernel Optimization that is GPU Calculation
>- fused Kernel
>- memory coalescing
>- theread/block optimization of batch

LLM Technic

#### a. Fused Kernel
Definition: Technology that combines multiple operations into one CUDA kernel (=fuse) and executes them at once.

*Typical Arithmetic Flow*
> 1. Kernel Launch
> 2. DATA LOAD by GPU
> 3. Arithmetic Execution
> 4. Save Results  
>
> So that, if has the multiple arithmetics run separate KERNELS.  

> Kernel1: x + b  
> Kernel2: ReLU(x)  
> Kernel3: Dropout(x)  

*Fused Kernel*
> Kernel1: (x + b -> ReLU -> Dropout)

In GPU, the most **EXPENSIVE** function is the `memory bandwidth`  
- Flash Attention also selected method that combining `QKᵀ -> Softmax -> V` for Minimizing fused kernel.

Examples:   

| Fused Kernel    | 설명                             |  
| --------------- | ------------------------------ |  
| Fused LayerNorm | LayerNorm + Add + Bias 등 합침    |  
| Fused MLP       | Linear + GeLU + Dropout 합침     |  
| Fused Attention | Q,K,V 계산 + softmax + matmul 합침 |  
| FlashAttention  | 완전한 Attention 패턴 퓨전        |  

#### b. Memory Coalescing


#### c. Thread/Block Optimization of Batch


### 2. Operation Optimization
Change of Model Structure
>- Flash Attention
>- Fused MLP
>- LayerNorm fusion
>- QKV 

Using Operation Library that Pytorch or NVIDIA

### 3. Model Compression
Model Compression 
>- Quantization
>- Pruning
>- Distilation

Descend Amount of Caputation

### 4. CUDA Tunnig and Optimization
Optimization GPU Code on CUDA LEVEL

>- NVIDIA Researcher LEVEL