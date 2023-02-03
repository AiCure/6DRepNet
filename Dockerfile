ARG AWS_ACCOUNT_ID=272510231547
ARG AWS_REGION=us-west-2
FROM python:3.8

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
# Above line may be a requirement for installing opencv-python
RUN apt-get install -y cmake --upgrade
RUN apt-get update && apt-get install -y ffmpeg python-tk

RUN pip install --upgrade pip
RUN pip install awscli

RUN aws configure set default.s3.signature_version s3v4
RUN aws configure set default.region us-west-2

RUN pip install pyarrow
# Needed for logging I think. Def need this one

RUN mkdir /hpose
WORKDIR /hpose
COPY requirements.txt .

COPY ./aicurelib ./aicurelib
COPY ./batch_base ./batch_base

RUN pip install -e ./aicurelib
RUN pip install -e ./batch_base

RUN pip install -r requirements.txt
COPY ./sixdrepnet ./sixdrepnet
COPY ./run_headpose.py ./run_headpose.py

ENTRYPOINT ["python", "run_headpose.py"]