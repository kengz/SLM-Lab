# run instructions:
# build image: docker build -t keng/slm_lab:latest -t keng/slm_lab:v2.0.0 .
# start container: docker run --name my_lab -dt keng/slm_lab
# enter container: docker exec -it my_lab bash
# remove container (forced): docker rm my_lab -f
# list image: docker images -a
# push image: docker push keng/slm_lab
# prune: docker system prune

FROM pytorch/pytorch:0.4_cuda9_cudnn7

LABEL maintainer="kengzwl@gmail.com"
LABEL website="https://github.com/kengz/SLM-Lab"

SHELL ["/bin/bash", "-c"]

# install system dependencies for OpenAI gym
RUN apt-get update && \
    apt-get install -y nano git cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev swig build-essential libstdc++6
RUN apt-get install -y python3-numpy python3-dev python3-pip python3-setuptools

RUN curl -sL https://deb.nodesource.com/setup_8.x | bash - && \
    apt-get install -y nodejs && \
    npm install -g yarn

RUN curl -O https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    rm -rf /opt/conda && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> /root/.bashrc && \
    conda update -n base conda

# install Python and Conda dependencies
RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda config --add channels conda-forge && \
    conda config --add channels pytorch && \
    conda create -n lab python=3.6 ipykernel -c conda-forge -c pytorch -y && \
    conda activate lab && \
    python -m ipykernel install --user --name lab

RUN echo "conda activate lab" >> /root/.bashrc

# create and set the working directory
RUN mkdir -p /opt/SLM-Lab

WORKDIR /opt/SLM-Lab

# install dependencies, only retrigger on dependency changes
COPY package.json package.json
RUN yarn install && \
    yarn global add electron@1.8.4 orca

COPY environment.yml environment.yml
RUN conda env update -f environment.yml

# Mac uses box2d-kengz, ubuntu uses box2d
RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda activate lab && \
    pip uninstall -y box2d-kengz && \
    pip install box2d

# copy file at last to not trigger changes above unnecessarily
COPY . .

RUN ./bin/copy_config

RUN . /opt/conda/etc/profile.d/conda.sh && \
    find . -name "__pycache__" -print0 | xargs -0 rm -rf && \
    find . -name "*.pyc" -print0 | xargs -0 rm -rf && \
    conda activate lab && \
    python setup.py test && \
    yarn reset

CMD ["/bin/bash"]
