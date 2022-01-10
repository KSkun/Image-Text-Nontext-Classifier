FROM ubuntu:focal

WORKDIR /app
COPY . .

# change apt source
RUN apt update && apt upgrade -y ca-certificates
COPY ./docker/sources.list /etc/apt/sources.list

# install pytorch
RUN apt update && apt install -y python3 python3-pip gcc g++ gfortran zlib1g-dev libjpeg-dev && rm -rf /var/lib/apt/lists/*
RUN pip install torch==1.10.1+cpu torchvision==0.11.2+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html

# install requirements
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /app/workdir
CMD [ "python3", "../src/main.py" ]
