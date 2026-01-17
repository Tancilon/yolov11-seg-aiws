FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 基础依赖
RUN apt-get update && apt-get install -y \
    curl bzip2 ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

# 安装 micromamba
RUN curl -fsSL https://micro.mamba.pm/api/micromamba/linux-64/latest -o /tmp/micromamba.tar.bz2 && \
    tar -xvjf /tmp/micromamba.tar.bz2 -C /usr/local/bin --strip-components=1 bin/micromamba && \
    /usr/local/bin/micromamba --version

ENV PATH=/usr/local/bin:$PATH

WORKDIR /app
COPY environment.yml /app/environment.yml

# 创建 conda 环境
RUN /usr/local/bin/micromamba create -y -n yolo11 -f /app/environment.yml && \
    /usr/local/bin/micromamba clean -a -y
ENV PATH=/opt/conda/envs/yolo11/bin:$PATH

# 拷贝代码（不包含数据集）
COPY . /app

# 默认推理命令（可覆盖）
CMD ["python", "-c", "from ultralytics import YOLO; YOLO('ckpt/yolo11n-seg.pt').predict('image.jpg', save=True)"]
