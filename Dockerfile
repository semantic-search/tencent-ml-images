FROM tensorflow/tensorflow:1.4.1-gpu
RUN apt-get update
RUN apt-get update --fix-missing
RUN apt-get -y upgrade
RUN apt-get install -y git
RUN apt-get install -y python-pip
RUN DEBIAN_FRONTEND="noninteractive" apt-get -y install tzdata
RUN apt-get -y install libopencv-dev python-opencv
RUN pip install python-dotenv kafka-python
RUN pip install redis
RUN pip install opencv-contrib-python
RUN git clone https://github.com/Tencent/tencent-ml-images.git
WORKDIR /notebooks/tencent-ml-images
COPY . .
CMD ["python", "main.py"]