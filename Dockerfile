# Use a specific Python version
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 拷贝代码到镜像中
COPY . .

# 安装依赖
RUN pip install --upgrade pip \
 && pip install -e .

RUN pip install -r requirements.txt

# Create a non-root user and switch to it
RUN useradd -m appuser
USER appuser

CMD ["bash"]