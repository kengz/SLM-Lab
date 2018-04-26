FROM continuumio/miniconda3 AS conda_container

FROM tensorflow/tensorflow:1.8.0-rc1-py3 AS tf_container

COPY --from=conda_container /opt/conda /opt/conda
COPY --from=conda_container /opt/conda/bin/conda /usr/local/bin/conda

# change default shell from sh to bash
SHELL ["/bin/bash", "-c"]

RUN curl -sL https://deb.nodesource.com/setup_8.x | bash - && \
    apt-get install -y nodejs && \
    npm install -g yarn

RUN node -v
RUN yarn -v

# # install system dependencies for OpenAI gym
# RUN apt-get update && \
#     apt-get install -y nano git python-numpy python-dev cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev swig
#
# # install Python and Conda dependencies
# RUN conda config --add channels conda-forge && \
#     conda config --add channels pytorch && \
#     conda create -n lab python=3.6 ipykernel -c conda-forge -c pytorch -y && \
#     echo "conda activate lab" >> /root/.bashrc
#
# RUN conda activate lab && python -m ipykernel install --user --name lab
#
# # Copy the current directory contents
# COPY . ~/SLM-Lab
#
# # Set the working directory to /app
# WORKDIR ~/SLM-Lab
#
# # install dependencies
# RUN yarn install
# RUN conda env update -f environment.yml
#
# RUN yarn test
# # # Run app.py when the container launches
# # CMD ["yarn test"]
#
# # docker image ls
# # docker container ls --all
# # docker system prune
# # docker build -t slm-lab .
# # docker run --name test1 -d slm-lab
# # docker exec -it test1 bash
# # docker stop test1
# # docker rm test1
#
# # ENTRYPOINT [ "/usr/bin/tini", "--" ]
# # CMD [ "/bin/bash" ]
