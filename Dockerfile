# 1️⃣ 베이스 이미지 선택 (Python 3.11)
FROM python:3.11-slim

# 2️⃣ uv 설치 (공식 이미지에서 바이너리만 복사하는 방식 추천)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# 3️⃣ 환경 변수 설정
# - UV_COMPILE_BYTECODE: 가져오기 속도 향상을 위해 바이트코드 컴파일
# - UV_LINK_MODE: Docker 내에서는 하드링크 대신 copy가 안정적일 수 있음
# - UV_PROJECT_ENVIRONMENT: 가상환경을 /app 내부가 아닌 안전한 곳에 생성
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT="/uv-venv"

# 4️⃣ 작업 디렉토리 설정
WORKDIR /app

# 5️⃣ 의존성 파일 복사 (pyproject.toml만 복사)
# 소스 코드가 바뀌어도 의존성이 안 바뀌면 이 단계는 캐시됨
COPY pyproject.toml ./

# ⭐️ 수정된 부분: uv.lock 파일은 복사하지 않고, 컨테이너 내부에서 생성 ⭐️

# 5️⃣ 프로젝트 의존성 설치
# RUN uv lock: pyproject.toml 기반으로 리눅스용 uv.lock을 새로 생성.
# RUN uv sync: 새로 생성된 uv.lock 기반으로 가상 환경에 설치.
RUN uv lock && \
    uv sync --no-install-project

# 6️⃣ 가상환경을 PATH에 추가 (uv run 없이도 python 실행 가능하게 함)
ENV PATH="$UV_PROJECT_ENVIRONMENT/bin:$PATH"

# 7️⃣ 프로젝트 파일 전체 복사
COPY . .

# 8️⃣ 프로젝트 자체 설치 (필요한 경우, 없다면 생략 가능)
# RUN uv sync --frozen

# 9️⃣ 실행 명령
# PATH에 venv가 등록되었으므로 바로 python 실행 가능
CMD ["python", "app.py"]