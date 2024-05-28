FROM python:3.10-slim

WORKDIR /app

ENV TZ="Asia/Shanghai"
ENV PYTHONPATH=/app
RUN echo "Asia/Shanghai" > /etc/timezone \
    && rm -rf /etc/localtime \
    && ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime

COPY requirements.txt requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY g_core g_core
COPY g_qbot g_qbot
COPY config config

ENTRYPOINT ["python", "g_qbot/main.py"]