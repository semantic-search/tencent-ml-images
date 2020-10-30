FROM tensorflow/tensorflow:1.4.1-gpu
RUN apt-get update
RUN apt-get update --fix-missing
RUN apt-get install -y git
RUN apt-get install -y python-pip
RUN DEBIAN_FRONTEND="noninteractive" apt-get -y install tzdata
RUN apt-get -y install libopencv-dev python-opencv
RUN git clone https://github.com/Tencent/tencent-ml-images.git
WORKDIR /notebooks/tencent-ml-images
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
COPY pylogbeat.py /usr/local/lib/python2.7/dist-packages/pylogbeat.py
CMD ["python", "main.py"]
