## GPU Acceleration
```
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```
### GPU Optimization
1. Kernel Optimization
2. Kernel Optimization
3. Operation Optimization

### 1. Kernel Optimization
CUDA Kernel Optimization that is GPU Calculation
>- fused Kernel
>- memory coalescing
>- theread/block optimization of batch

LLM Technic

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