# 1️⃣ 베이스 이미지 선택
FROM python:3.11

# 2️⃣ 작업 디렉토리 설정
WORKDIR /app

# 3️⃣ uv 설치
RUN pip install --upgrade pip \
    && pip install uv

# 4️⃣ 프로젝트 파일 복사 (main.py, requirements.txt 등)
COPY . .

# 5️⃣ uv로 패키지 설치
# 이미 requirements.txt나 uv.lock이 있다면
RUN uv sync

# 6️⃣ 컨테이너 시작 시 Python 스크립트 실행
CMD ["uv", "run", "python", "main.py"]
