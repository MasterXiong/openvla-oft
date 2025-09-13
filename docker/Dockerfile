# Base Image
ARG PARENT_IMAGE
FROM $PARENT_IMAGE

# Args need to be below FROM!
ARG USER_ID
ARG GROUP_ID
ARG PYTHON_VERSION=3.10

ENV NVIDIA_DRIVER_CAPABILITIES all

# Install os-level packages
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    bash-completion \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    git \
    git-lfs \
    htop \
    libegl1 \
    libxext6 \
    libjpeg-dev \
    libpng-dev  \
    libvulkan1 \
    ffmpeg \
    rsync \
    tmux \
    unzip \
    vim \
    vulkan-utils \
    wget \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# Install (mini) conda
RUN curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda init && \
    /opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    /opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    /opt/conda/bin/conda install -y python="$PYTHON_VERSION" && \
    /opt/conda/bin/conda clean -ya

ENV PATH /opt/conda/bin:$PATH
SHELL ["/bin/bash", "-c"]

# install openvla-oft
RUN pip install --upgrade pip
# RUN pip install torch torchvision torchaudio
COPY ./ /user/openvla-oft
WORKDIR /user/openvla-oft
RUN pip install -e .
# Install Flash Attention 2 for training (https://github.com/Dao-AILab/flash-attention)
#   =>> If you run into difficulty, try `pip cache remove flash_attn` first
RUN pip install packaging ninja
# ninja --version; echo $?  # Verify Ninja --> should return exit code "0"
RUN pip install "flash-attn==2.5.5" --no-build-isolation
# fix incompitable package versions
# https://github.com/moojink/openvla-oft/issues/97
RUN pip install opencv-python==4.10.0.84 numpy==1.26.4

# install LIBERO
RUN pip install -r experiments/robot/libero/libero_requirements.txt
WORKDIR /
RUN git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
WORKDIR /LIBERO
RUN pip install -e .

# Change permissions
WORKDIR /user/openvla-oft
RUN groupadd -g ${GROUP_ID} user
RUN useradd --shell /bin/bash -u ${USER_ID} -g ${GROUP_ID} -o -d /user user
RUN chown -R user:user /user && chmod -R u+w /user
# RUN chown -R user:user / && chmod -R u+w /

# Set python ENV variables
ENV PYTHONUNBUFFERED=1