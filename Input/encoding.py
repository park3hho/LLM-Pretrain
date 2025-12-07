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

# 5) 로드 시도 (strict=True -> 실패하면 strict=False로 재시도하여 missing/unexpected 키 출력)
try:
    res = model.load_state_dict(new_sd, strict=True)
    print("Loaded with strict=True. missing_keys:", res.missing_keys, "unexpected_keys:", res.unexpected_keys)
except Exception as e:
    print("strict=True failed:", e)
    res = model.load_state_dict(new_sd, strict=False)
    print("Loaded with strict=False. missing_keys:", res.missing_keys, "unexpected_keys:", res.unexpected_keys)

# 6) 모델을 eval()로 전환
model.eval()
print("Model loaded and set to eval() on device:", device)
