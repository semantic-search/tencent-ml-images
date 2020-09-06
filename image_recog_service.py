import numpy as np
import cv2 as cv
import tensorflow as tf
from models import resnet as resnet
import os


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


def preprocess(img):
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
    img = img[int((newH-test_crop)/2):int((newH-test_crop)/2)+int(test_crop), int((newW-test_crop)/2):int((newW-test_crop)/2)+int(test_crop)]
    img = ((img/255.0) - 0.5) * 2.0
    img = img[..., ::-1]
    return img


def predict(file_name, doc=False):

    # inference
    labels = []
    scores_np = []
    scores = []
    raw_img = cv.imread(file_name)
    img = preprocess(raw_img)
    logits, probs_topk, preds_topk = sess.run(
        [logit, prob_topk, pred_topk],
        {images: np.expand_dims(img, axis=0)})
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
    if doc:
        full_res = {
            "labels": labels,
            "scores": scores
        }
        text_resp = {
            "labels": labels
        }
        response_dict = {
            "full_results": full_res,
            "text": text_resp
        }
        os.remove(file_name)
        return response_dict
    else:
        full_res = {
            "file_name": file_name,
            "labels": labels,
            "scores": scores,
            "is_doc_type": False

        }
        text_resp = {
            "file_name": file_name,
            "labels": labels,
            "is_doc_type": False
        }
        response_dict = {
            "full_results": full_res,
            "text": text_resp
        }
        os.remove(file_name)
        return response_dict
