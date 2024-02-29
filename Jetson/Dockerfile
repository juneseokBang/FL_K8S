FROM nvcr.io/nvidia/l4t-pytorch:r32.6.1-pth1.9-py3

# 시스템 패키지 업데이트 및 필요한 패키지 설치
RUN apt-get update && apt-get install -y \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Python 패키지 업데이트 및 Flask 설치
RUN pip3 install --upgrade pip && \
    pip3 install Flask && \
    pip3 install requests

# 모델 스크립트 복사
COPY model_train_test.py /model_train_test.py
COPY fl_client.py /fl_client.py

# 기본 명령어 설정
CMD ["python3", "/model_train_test.py"]
