FROM continuumio/anaconda3
MAINTAINER Simon Moulds "simon.moulds@imperial.ac.uk"

RUN apt-get update && apt-get install -y gfortran

RUN /opt/conda/bin/conda update -n base -c defaults conda && \
    /opt/conda/bin/conda install python=3 && \
    /opt/conda/bin/conda install pip

WORKDIR /
COPY . /

RUN python -m pip install -e .


WORKDIR /app
ENTRYPOINT ["aquacrop"]