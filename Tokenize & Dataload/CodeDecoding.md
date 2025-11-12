# [CD] Multi-Head Attention
Transformer의 핵심 구성요소인 “Multi-Head Attention” (다중 헤드 어텐션) 을 PyTorch로 직접 구현한 클래스.

## 🧩 전체 구조 요약
```
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out):  # 초기화
        ...
    def forward(self, x):  # 순전파
        ...
```
Transformer에서 입력 x (예: 토큰 임베딩)을 받아
“단어 간의 연관성(Attention)”을 계산하고,
그 결과를 다시 임베딩 형태로 돌려주는 모듈

## 클래스 정의 및 초기화 부문
### 인자
``` 인자
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
```
- `d_in`: 입력 차원 (입력 벡터의 feature 수)
- `d_out`: 출력 차원 (전체 헤드들을 합친 차원)
- `super().__init__()`로 `nn.Module` 초기화.

> nn.Module이 뭔데?
>- nn.Module은 PyTorch에서 신경망(Neural Network)을 만들 때 사용하는 **모든 모델의 기본 클래스(base class)**
### 헤드 개수 할당
``` 
assert d_out % NUM_HEADS == 0, "d_out must be divisible by n_heads"
```
- d_out이 NUM_HEADS로 나누어 떨어지는지 확인. 각 헤드의 차원(head_dim) = d_out // NUM_HEADS.
- d_out은 출력 차원 수 (예: 512)
- NUM_HEADS는 어텐션 헤드의 개수 (예: 8)
- 각 헤드가 처리할 차원을 동일하게 나누기 위해 d_out이 NUM_HEADS로 나누어떨어져야 함
- 컴퓨팅 자원 여유가 있으면 크게.

### Dimension
``` dim
self.d_out = d_out
self.head_dim = d_out // NUM_HEADS
```
- 내부 저장. `head_dim`은 한 헤드가 가지는 feature 수
- 하나의 어텐션 헤드가 담당하는 차원 크기
- 예: d_out=512, NUM_HEADS=8 → head_dim=64

### QKV 가중치 행렬
``` Weight Matrix
self.W_query = nn.Linear(d_in, d_out, bias=QKV_BIAS)
self.W_key = nn.Linear(d_in, d_out, bias=QKV_BIAS)
self.W_value = nn.Linear(d_in, d_out, bias=QKV_BIAS)
```
- 입력 벡터 x를 각각 Query, Key, Value로 변환하는 선형 변환 (가중치 행렬).
- 즉, 입력 문장의 각 단어를 3가지 역할로 매핑하는 과정.
- `QKV_BIAS`는 전역 상수로 bias 사용 여부(참/거짓)

### 선형 
``` out_proj, Dropout
self.out_proj = nn.Linear(d_out, d_out)
self.dropout = nn.Dropout(DROP_RATE)
```
- 여러 헤드의 출력을 다시 합쳐 최종 출력으로 변환 / 여러 헤드를 합친 결과에
마지막으로 적용되는 출력 투사층 
- Dropout으로 일부 연결을 끊어 과적합 방지(regularization)
> regularization이 왜 필요한데? `d_model`이 무슨 의미인데?

### 
``` 
self.register_buffer('mask', torch.triu(torch.ones(CONTEXT_LENGTH, CONTEXT_LENGTH), diagonal=1))
```
- 상삼각행렬(위쪽이 1, 아래는 0)
- “미래 토큰을 보지 못하게” 하는 캐주얼 마스크 (causal mask) (GPT 같은 언어모델에서 중요)

예시 (길이 4):
[[0, 1, 1, 1],
 [0, 0, 1, 1],
 [0, 0, 0, 1],
 [0, 0, 0, 0]]

## 순전파 (foward)
(1) 입력 형태
b, num_tokens, d_in = x.shape


b: batch size

num_tokens: 한 문장(시퀀스)의 토큰 수

d_in: 입력 임베딩 차원

예: (b=16, num_tokens=128, d_in=512)

(2) Q, K, V 계산
keys = self.W_key(x)
queries = self.W_query(x)
values = self.W_value(x)


입력 문장의 각 단어를
Query, Key, Value 벡터로 매핑합니다.
shape: (b, num_tokens, d_out)

(3) 여러 헤드로 분리
keys = keys.view(b, num_tokens, NUM_HEADS, self.head_dim)
queries = queries.view(b, num_tokens, NUM_HEADS, self.head_dim)
values = values.view(b, num_tokens, NUM_HEADS, self.head_dim)


이제 각 단어의 임베딩을 NUM_HEADS 개로 쪼갭니다.

원래: (b, num_tokens, 512)

바뀐 후: (b, num_tokens, 8, 64)

(4) 차원 재배열 (헤드 기준 계산하기 위해)
keys = keys.transpose(1, 2)
queries = queries.transpose(1, 2)
values = values.transpose(1, 2)


→ (b, NUM_HEADS, num_tokens, head_dim)

이제 각 헤드별로 어텐션 계산 가능.

(5) 어텐션 스코어 계산
attn_scores = queries @ keys.transpose(2, 3)


이 연산은 Q × Kᵀ (행렬 곱) 입니다.

각 토큰이 다른 토큰과 얼마나 연관되는지를 나타냄.

shape: (b, NUM_HEADS, num_tokens, num_tokens)

(6) 마스크 적용 (미래 정보 차단)
mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
attn_scores.masked_fill_(mask_bool, -torch.inf)


마스크가 1인 위치(=미래)는 -inf로 채워
softmax 후 0이 되게 함.

즉, 현재 단어는 미래 단어를 볼 수 없음.

(7) Softmax로 가중치 변환
```
attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
```
스코어를 정규화하여 확률처럼 만듭니다.

√dₖ로 나누는 이유: 큰 차원일수록 내적값이 커져
softmax가 saturation 되는 걸 방지.

(8) Value를 가중합
```
context_vec = (attn_weights @ values).transpose(1, 2)
```

각 토큰의 “문맥 정보”를 계산.

(b, num_tokens, NUM_HEADS, head_dim)

(9) 여러 헤드 결과 합치기
```
context_vec = context_vec.reshape(b, num_tokens, self.d_out)
context_vec = self.out_proj(context_vec)
```

헤드별 출력을 합쳐 원래 차원으로 되돌림.

최종 출력 shape: (b, num_tokens, d_out)