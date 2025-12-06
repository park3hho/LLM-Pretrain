import re
import matplotlib.pyplot as plt

# 로그 파일 읽기
with open("/app/Tokenize & Dataload/Epoch/train_log.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Avg Loss 값만 정규식으로 추출
losses = [float(x) for x in re.findall(r"Avg\s+Loss:\s*([0-9\.eE+-]+)", text)]

# 그래프 그리기
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Avg Loss")
plt.title("Training Loss")
plt.show()
plt.savefig("loss.png")   # 🔥 컨테이너 내부에 loss.png 저장