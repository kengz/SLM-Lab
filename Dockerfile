FROM floydhub/pytorch:0.3.0-py3.22 AS pytorch_container

LABEL maintainer="kengzwl@gmail.com"
LABEL website="https://github.com/kengz/SLM-Lab"

SHELL ["/bin/bash", "-c"]

# install system dependencies for OpenAI gym
RUN apt-get update && \
    apt-get install -y nano git python-numpy python-dev cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev swig

RUN curl -sL https://deb.nodesource.com/setup_8.x | bash - && \
    apt-get install -y nodejs && \
    npm install -g yarn

ENV PATH /opt/conda/bin:$PATH

RUN curl -O https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> /root/.bashrc && \
    conda update -n base conda

# install Python and Conda dependencies
RUN conda config --add channels conda-forge && \
    conda config --add channels pytorch && \
    conda create -n lab python=3.6 ipykernel -c conda-forge -c pytorch -y && \
    source activate lab && \
    python -m ipykernel install --user --name lab

RUN echo "source activate lab" >> /root/.bashrc

# create and set the working directory
RUN mkdir -p /opt/SLM-Lab

WORKDIR /opt/SLM-Lab

# install dependencies, only retrigger on dependency changes
COPY package.json package.json
RUN yarn install

COPY environment.yml environment.yml
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64
RUN conda env update -f environment.yml

# copy file at last to not trigger changes above unnecessarily
COPY . .

RUN source activate lab && \
    yarn test

CMD ["/bin/bash"]
