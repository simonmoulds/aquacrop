FROM ubuntu:20.04
MAINTAINER Simon Moulds "simon.moulds@imperial.ac.uk"

RUN apt-get update -y && \
    apt-get install -y python3-pip python3-dev

COPY . /app
WORKDIR /app

RUN pip3 install --upgrade pip && \
    pip3 install -r requirements.txt

RUN pip3 install -e .

CMD [ "aquacrop" ]