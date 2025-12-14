print("\n============================Default_Setting==========================")
import sys
import os
import torch
from collections import OrderedDict

# 프로젝트 루트를 import 경로에 추가 (필요시)
sys.path.append(os.path.abspath(".."))

# 모델 정의 import (패키지/경로가 맞는지 확인)
from Tokenize_Dataload.NeuralNetworkModel import GPTModel

# 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 체크포인트 경로
CKPT = "../Tokenize_Dataload/Epoch/train_results/model_020.pth"

# 1) 먼저 파일을 한 번 로드해서 타입/키 확인 (디버그용 출력)
ck = torch.load(CKPT, map_location="cpu")
print("\n===========================Load_Checkpoint==========================")
print("Loaded checkpoint type:", type(ck))
if isinstance(ck, dict):
    print("Checkpoint keys (first 30):", list(ck.keys())[:30])

# 2) 모델 인스턴스 생성 (네 GPTModel __init__ 에 인자가 필요하면 () 안에 넣을 것)
model = GPTModel()          # <-- 필요시 GPTModel(파라미터들...) 로 수정
model.to(device)

# 3) state_dict 결정 (checkpoint가 dict인지, 직접 state_dict인지 검사)
if isinstance(ck, dict):
    # 보편적인 키 이름들 처리
    if "model_state_dict" in ck:
        state_dict = ck["model_state_dict"]
    elif "state_dict" in ck:
        state_dict = ck["state_dict"]
    else:
        # 이미 OrderedDict(state_dict) 형태로 저장된 케이스일 수 있음
        state_dict = ck
else:
    state_dict = ck

# 4) 'module.' prefix 제거(DDP에서 저장된 경우 안전 장치)
new_sd = OrderedDict()
for k, v in state_dict.items():
    new_k = k
    if k.startswith("module."):
        new_k = k[len("module."):]
    new_sd[new_k] = v

print("\n============================state_dict()============================")
# 5) 로드 시도 (strict=True -> 실패하면 strict=False로 재시도하여 missing/unexpected 키 출력)
try:
    res = model.load_state_dict(new_sd, strict=True)
    print("Loaded with strict=True. missing_keys:", res.missing_keys, "unexpected_keys:", res.unexpected_keys)
except Exception as e:
    print("strict=True failed:", e)
    res = model.load_state_dict(new_sd, strict=False)
    print("Loaded with strict=False. missing_keys:", res.missing_keys, "unexpected_keys:", res.unexpected_keys)

print("\n================================eval()==============================")
# 6) 모델을 eval()로 전환
model.eval()
print("Model loaded and set to eval() on device:", device)

print("\n========================Most_related_words=========================")
import tiktoken # pip install tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
Encoded_Word = "Dobby is"
idx = tokenizer.encode(f"{Encoded_Word}") # 토큰 id의 list
print(f"Encoded Word: {Encoded_Word}")
print(f"Tokenized Result: {idx}\n")
idx = torch.tensor(idx).unsqueeze(0).to(device)

with torch.no_grad():
    logits = model(idx)

logits = logits[:, -1, :]

# 가장 확률이 높은 단어 10개 출력
top_logits, top_indices = torch.topk(logits, 10)
for p, i in zip(top_logits.squeeze(0).tolist(), top_indices.squeeze(0).tolist()):
    print(f"{p:.2f}\t {i}\t {tokenizer.decode([i])}")

# 가장 확률이 높은 단어 출력
idx_next = torch.argmax(logits, dim=-1, keepdim=True)
flat = idx_next.squeeze(0) # 배치 차원 제거 torch.Size([1])
out = tokenizer.decode(flat.tolist()) # 텐서를 리스트로 바꿔서 디코드
print(out)


def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)

        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        if idx_next == eos_id:
            break

        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

    return idx

start_context = input("Start context: ")

# idx = tokenizer.encode(start_context, allowed_special={'<|endoftext|>'})
idx = tokenizer.encode(start_context)
idx = torch.tensor(idx).unsqueeze(0)

context_size = model.pos_emb.weight.shape[0]

for i in range(10):
    token_ids = generate(
        model=model,
        idx=idx.to(device),
        max_new_tokens=50,
        context_size= context_size,
        top_k=50,
        temperature=0.5
    )

    flat = token_ids.squeeze(0) # remove batch dimension
    out = tokenizer.decode(flat.tolist()).replace("\n", " ")

    print(i, ":", out)