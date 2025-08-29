# run instructions:
# build image: docker build -t kengz/slm_lab:latest -t kengz/slm_lab:v4.3.0 .
# start container: docker run --rm -it kengz/slm_lab:v4.3.0
# list image: docker images -a
# push image: docker push kengz/slm_lab
# prune: docker system prune

FROM ubuntu:22.04

LABEL maintainer="kengzwl@gmail.com"
LABEL website="https://github.com/kengz/SLM-Lab"

SHELL ["/bin/bash", "-c"]

RUN apt-get update && \
    apt-get install -y build-essential \
    curl nano git wget zip libstdc++6 \
    python3-dev zlib1g-dev libjpeg-dev cmake swig python-pyglet python3-opengl libboost-all-dev libsdl2-dev libosmesa6-dev patchelf ffmpeg xvfb && \
    rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc

# create and set the working directory
RUN mkdir -p /root/SLM-Lab

WORKDIR /root/SLM-Lab

# install dependencies, only retrigger on dependency changes
COPY pyproject.toml uv.lock ./

# install Python dependencies with uv
RUN export PATH="$HOME/.local/bin:$PATH" && \
    uv sync --frozen

# copy file at last to not trigger changes above unnecessarily
COPY . .

RUN export PATH="$HOME/.local/bin:$PATH" && \
    uv run python -m pytest --verbose --no-flaky-report test/

CMD ["/bin/bash"]
