FROM tensorflow/tensorflow:1.8.0-rc1-py3

# change default shell from sh to bash
SHELL ["/bin/bash", "-c"]

# install system dependencies for OpenAI gym
RUN apt-get update && \
    apt-get install -y nano git python-numpy python-dev cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev swig

# Install node
RUN curl -sL https://deb.nodesource.com/setup_8.x | bash - && \
    apt-get install -y nodejs && \
    npm install -g yarn

# Install conda and python dependencies
RUN curl --silent -O https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh

RUN bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh && \
    ln -s /opt/conda/bin/conda /usr/local/bin/conda && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> /root/.bashrc

RUN conda config --add channels conda-forge && \
    conda config --add channels pytorch && \
    conda create -n lab python=3.6 ipykernel -c conda-forge -c pytorch -y && \
    echo "conda activate lab" >> /root/.bashrc

RUN conda activate lab && \
    python -m ipykernel install --user --name lab

# # Copy the current directory contents
# COPY . ~/SLM-Lab
#
# # Set the working directory to /app
# WORKDIR ~/SLM-Lab
#
# RUN pwd
# RUN conda env update -f environment.yml
# RUN yarn install
#
#
#
#
# # Make port 80 available to the world outside this container
# # EXPOSE 80
#
# # Run app.py when the container launches
# CMD ["yarn test"]

# docker image ls
# docker container ls --all
# docker system prune
# docker build -t test .
# docker run --name test1 -d test
# docker exec -it test1 bash
# docker stop test1
# docker rm test1
