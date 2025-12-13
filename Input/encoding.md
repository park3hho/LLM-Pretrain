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
Definition: When Number of Threads read a memory, make memory `sequential` and `coalesce` them.

`WARP`: 32 threads
>- 1:1 Mapping among Threads-Index and Memory-Index 
>- row-reading
>- Minimize Stride
>- Pre-Transpose
>- shared-memory(Other Way to Reduce GPU-load)
>- Triton(Making Sequential Memory Address)
>- Pytorch AutoCoalescing

*1:1 Mapping among Threads-Index and Memor-Index*
```aiignore
int idx = threadIdx.x + blockDim.x * blockIdx.x;
output[idx] = input[idx];
```
→ thread0 → input[0]  
→ thread1 → input[1]  
→ thread2 → input[2]  
→ …

*row-reading*
PyTorch, Numpy: row-major

Good Example
```(coalescing O)
float val = A[row][threadIdx.x];  // 행에서 연속된 원소 읽기
```
Bad Example
```(coalescing X)
float val = A[threadIdx.x][col];  // 열을 따라 접근 → 주소가 뜬금없이 멀어짐
```

#### c. Thread/Block Optimization of Batch
Definition: Batch dimension(B) / what thread handle / what block bind / assign how many thread 

### 2. Operation Optimization
Change of Model Structure
>- Flash Attention
>- Fused MLP
>- LayerNorm fusion
>- QKV

Using Operation Library that Pytorch or NVIDIA

#### Flash Attetntion
Problem of Attention Model
```
Q @ K^T -> (b, heads, seq, seq)
```
If length of seq is too long, memory cannot hold it.

*The Idea of Flash Attention*
> Do not make Length of Attention to n^2
> Cut units of the tile, as soon as calculate and abandon it.


#### Fused MLP, LayerNorm
Normal MLP Sequence
```
Linear -> GELU -> Linear -> Dropout
```
in Kernel
```
Kernel Calling 1: Linear
Kernel Calling 2: GELU
Kernel Calling 3: Linear
Kernel Calling 4: Dropout
```

*Fused Kernel*
```Fused kernel
W1 @ x -> GELU -> W2 @ (Result) -> Dropout
```
- use one "CUDA KERNEL"
- Same Logic with LayerNorm

#### QKV Fusion (QKV Projection Fusion)
Basic of Attention
```
Q = x * Wq  
K = x * Wk  
V = x * Wv
```
Three times of Kernel Calling

*QKV Fusion*
```perl
W = [Wq | Wk | Wv]
```
```Arithmetic Flow
QKV = x @ W
```

### 3. Model Compression
Model Compression 
>- Quantization
>- Pruning
>- Distilation

Descend Amount of Caputation

#### Quantization
float(32) -> INT8 or INT4
> Post-Training Quantization(PTQ): Quantizing After Training
> Quantization-Aware Training(QAT): Qunatizing During Training 

#### Pruning
Get rid of Unnecessary Parameters

#### Distilation
Teacher - Student
> Student Imitate Teacher's Answer

### 4. CUDA Tunnig and Optimization
Optimization GPU Code on CUDA LEVEL
>- a. Memory Hierarchy Optimization
>- b. Thread / Warp / Block Mapping
>- c. Instruction-Level Optimization
>- d. Asynchronous Execution & Overlap
>- e. Precision & Tensor Core Optimization

#### a. Memory Hierarchy Optimization
Main Key of GPU Calculation is Memory, not Arithmetic.

*Main Strategy*
- Global Memory: Access Minimization
- Shared / Register: Maximum Usage 

*Main Methods*
- Coalesced Memory Access
- Shared Memory tiling
- Register blocking
- Avoid bank conflict
- Prefetching

🔑 70% of CUDA Optimization up to Memory.

#### b. Thread / Warp / Block Mapping
(Hardware-Friendly Parallelization)
Thread(32) → Warp
a Number of Warp → SM

*Main Strategy*
- Warp divergence Reduction
- Occupancy Maximization

*Main Methods*
- Branch 제거 (if → mask)
- Thread-per-element Design
- Block size Tunning (128 / 256 / 512)
- Warp-specialization

#### c. Instruction-Level Optimization
Make Computing-Cost Cheaper

*Main Strategy*
- Reduction of Expensive Computing
- Pipelining

*Main Methods*
- FMA 
- fast math (__expf, __logf)
- Loop unrolling
- Instruction fusion

#### d. Asynchronous Execution & Overlap
*Main Strategy*  
- Arithmetic and Memory Overlap  

*Main Methods*
- CUDA Streams
- Async memcpy
- Double buffering
- Pipeline parallelism

#### e. Precision & Tensor Core Optimization
Hardware-Unit

(NVIDIA RESEARCHER MOST-NOTICED PARTS)

*Main Strategy*
- Usage of Tensor Core

*Main Methods*
- FP16/BF16
- INT8/INT4
- MMA Instruction
- Proper memory Aligning
