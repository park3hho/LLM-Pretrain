# Encoding Code Decoding
> a. GPU Accelerartion
> b. Declair of REASONING
> c. Tokenizer & Ready to Answer
> d. Single Token Prediction

## a. GPU Acceleration
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


## b. Declair of REASONING
```
model.eval()
```

## c. Tokenizer & Ready to Answer
```
import tiktoken # pip install tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
```
- OpenAI GPT-2와 동일한 BPE 토크나이저
- 모델이 학습될 때 사용한 토큰 분할 방식과 반드시 같아야 함
👉 토크나이저 다르면 출력은 전부 쓰레기

### Readiness
```
idx = torch.tensor(idx).unsqueeze(0).to(device)
```
| 코드                  | 의미                           |
| ------------------- | ---------------------------- |
| `torch.tensor(idx)` | 리스트 → 텐서                     |
| `unsqueeze(0)`      | batch 차원 추가 → `(1, seq_len)` |
| `.to(device)`       | CPU or GPU 이동                |

👉 모델 입력 형태 = (batch, sequence)

## d. Single Token Prediction
```
with torch.no_grad():
    logits = model(idx)
```
- 추론이므로 gradient 계산 X  
- 출력 형태:
```
(batch, seq_len, vocab_size)
```
---
```
logits = logits[:, -1, :]
```
- 마지막 토큰 기준으로 다음 토큰 확률만 사용
- shape:
```
(1, vocab_size)
```
---
🔝 Top-10 후보 출력
```
top_logits, top_indices = torch.topk(logits, 10)
```
확률(정확히는 logit)이 가장 높은 토큰 10개

---
```
for p, i in zip(top_logits.squeeze(0).tolist(), top_indices.squeeze(0).tolist()):
    print(f"{p:.2f}\t {i}\t {tokenizer.decode([i])}")
```
- logit 값
- 토큰 ID
- 사람이 읽을 수 있는 문자열

👉 모델이 “다음에 나올 것 같다”고 생각하는 단어들

---
```
idx_next = torch.argmax(logits, dim=-1, keepdim=True)
```
- greedy decoding  
- 가장 높은 확률 하나 선택  
---

```
flat = idx_next.squeeze(0)
out = tokenizer.decode(flat.tolist())
print(out)
```
- 텐서 → 문자열
- “Dobby is ___” 의 ___에 들어갈 단어

## e. 