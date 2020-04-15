# run instructions:
# build image: docker build -t kengz/slm_lab:latest -t kengz/slm_lab:v4.2.0 .
# start container: docker run --rm -it kengz/slm_lab:v4.2.0
# list image: docker images -a
# push image: docker push kengz/slm_lab
# prune: docker system prune

FROM ubuntu:16.04

LABEL maintainer="kengzwl@gmail.com"
LABEL website="https://github.com/kengz/SLM-Lab"

SHELL ["/bin/bash", "-c"]

RUN apt-get update && \
    apt-get install -y build-essential \
    curl nano git wget zip libstdc++6 \
    python3-dev zlib1g-dev libjpeg-dev cmake swig python-pyglet python3-opengl libboost-all-dev libsdl2-dev libosmesa6-dev patchelf ffmpeg xvfb && \
    rm -rf /var/lib/apt/lists/*

RUN curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b && \
    rm Miniconda3-latest-Linux-x86_64.sh && \
    echo '. ~/miniconda3/etc/profile.d/conda.sh' >> ~/.bashrc && \
    . ~/miniconda3/etc/profile.d/conda.sh && \
    conda --version

# create and set the working directory
RUN mkdir -p /root/SLM-Lab

WORKDIR /root/SLM-Lab

# install dependencies, only retrigger on dependency changes
COPY environment.yml environment.yml

# install Python and Conda dependencies
RUN . ~/miniconda3/etc/profile.d/conda.sh && \
    conda create -n lab python=3.7.3 -y && \
    conda activate lab && \
    conda env update -f environment.yml && \
    conda clean -y --all && \
    rm -rf ~/.cache/pip

# copy file at last to not trigger changes above unnecessarily
COPY . .

RUN . ~/miniconda3/etc/profile.d/conda.sh && \
    conda activate lab && \
    python setup.py test
    # pytest --verbose --no-flaky-report test/spec/test_dist_spec.py && \
    # yarn reset

CMD ["/bin/bash"]
