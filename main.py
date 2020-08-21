#!/usr/bin/python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import numpy as np
import cv2 as cv
import tensorflow as tf
from models import resnet as resnet
from flags import FLAGS
import json
from kafka import KafkaConsumer
from kafka import KafkaProducer
from json import loads
from base64 import decodestring
import base64
from dotenv import load_dotenv
import uuid
import redis
import sys

load_dotenv()

TOPIC = "TENCENT_ML_IMAGES"
KAFKA_HOSTNAME = os.getenv("KAFKA_HOSTNAME")
KAFKA_PORT = os.getenv("KAFKA_PORT")
RECEIVE_TOPIC = 'TENCENT_ML_IMAGES'
SEND_TOPIC_FULL = "IMAGE_RESULTS"
SEND_TOPIC_TEXT = "TEXT"
print("kafka : "+KAFKA_HOSTNAME+':'+KAFKA_PORT)
REDIS_HOSTNAME = os.getenv("REDIS_HOSTNAME")
REDIS_PORT = os.getenv("REDIS_PORT")
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")

r = redis.StrictRedis(host=REDIS_HOSTNAME, port=REDIS_PORT,
                      password=REDIS_PASSWORD, ssl=True)

r.set(TOPIC, "FREE")
consumer_tencent_ml_images = KafkaConsumer(
    RECEIVE_TOPIC,
    bootstrap_servers=[KAFKA_HOSTNAME+':'+KAFKA_PORT],
    auto_offset_reset="earliest",
    enable_auto_commit=True,
    group_id="my-group",
    value_deserializer=lambda x: loads(x.decode("utf-8")),
)

# For Sending processed img data further
producer = KafkaProducer(
    bootstrap_servers=[KAFKA_HOSTNAME+':'+KAFKA_PORT],
    value_serializer=lambda x: json.dumps(x).encode("utf-8"),
)



def _load_dictionary(dict_file):
    dictionary = dict()
    with open(dict_file, 'r') as lines:
        for line in lines:
            sp = line.rstrip('\n').split('\t')
            idx, name = sp[0], sp[1]
            dictionary[idx] = name
    return dictionary

# build model
images = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3])
net = resnet.ResNet(images, is_training=False)
net.build_model()

logit = net.logit
prob = tf.nn.softmax(logit)
prob_topk, pred_topk = tf.nn.top_k(prob, k=5)

# restore model
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# config.gpu_options.visible_device_list = 0
config.log_device_placement=False
sess = tf.Session(config=config)
saver = tf.train.Saver(tf.global_variables())
saver.restore(sess, 'checkpoints/resnet.ckpt')
dictionary = _load_dictionary('data/imagenet2012_dictionary.txt')

class Predict:
    def preprocess(self, img):
        rawH = float(img.shape[0])
        rawW = float(img.shape[1])
        newH = 256.0
        newW = 256.0
        test_crop = 224.0

        if rawH <= rawW:
            newW = (rawW/rawH) * newH
        else:
            newH = (rawH/rawW) * newW
        img = cv.resize(img, (int(newW), int(newH)))
        img = img[int((newH-test_crop)/2):int((newH-test_crop)/2)+int(test_crop),int((newW-test_crop)/2):int((newW-test_crop)/2)+int(test_crop)]
        img = ((img/255.0) - 0.5) * 2.0
        img = img[...,::-1]
        return img
    def recog(self, image_id, file_name):

        # inference
        labels = []
        scores_np = []
        scores = []
        raw_img = cv.imread(file_name)
        if type(raw_img)==None or raw_img.data==None :
            print("open pic " + os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename)) + " failed")
        img = self.preprocess(raw_img)
        logits, probs_topk, preds_topk = sess.run([logit, prob_topk, pred_topk],
            {images:np.expand_dims(img, axis=0)})
        probs_topk = np.squeeze(probs_topk)
        preds_topk = np.squeeze(preds_topk)
        names_topk = [dictionary[str(i)] for i in preds_topk]
        for i, pred in enumerate(preds_topk):
            labels.append(names_topk[i])
            scores_np.append(probs_topk[i])
        np.array(scores_np).tolist()
        for vals in scores_np:
            c = float(vals)
            scores.append(c)
        full_res = {
        "image_id": image_id,
        "labels": labels,
        "scores": scores
        }
        text_resp = full_res["labels"]
        response_dict = {
            "full_results": full_res,
            "text": text_resp
        }
        os.remove(file_name)
        return response_dict


if __name__ == '__main__':
    print("in main")
    for message in consumer_tencent_ml_images:
        print('xxx--- inside tencent ml images---xxx')
        print(KAFKA_HOSTNAME + ':' + KAFKA_PORT)
        message = message.value
        print("MESSAGE RECEIVED consumer_tencent_ml_images: ")
        image_id = message['image_id']
        data = message['data']
        r.set(RECEIVE_TOPIC, image_id)
        image_file = str(uuid.uuid4()) + ".jpg"
        with open(image_file, "wb") as fh:
            fh.write(base64.b64decode(data.encode("ascii")))
        predict_obj = Predict()
        response = predict_obj.recog(image_id=image_id, file_name=image_file)
        full_results_dict = response["full_results"]
        text_results = response["text"]
        # sending full and text res(without cordinates or probability) to kafka
        producer.send(SEND_TOPIC_FULL, value=json.dumps(full_results_dict))
        producer.send(SEND_TOPIC_TEXT, value=json.dumps(text_results))
        producer.flush()

