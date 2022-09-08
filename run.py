# coding=utf-8
import os
import shutil
import sys
import time
from configparser import ConfigParser
import cv2
import numpy as np
import tensorflow as tf
from ctpn.nets import model_train as model
from ctpn.ctpnutils.rpn_msr.proposal_layer import proposal_layer
from ctpn.ctpnutils.text_connector.detectors import TextDetector

import torch
from torch.autograd import Variable
from crnn import utils
from crnn import dataset
from PIL import Image
import models.crnn as crnn


con = configparser.ConfigParser()
con.read(file, encoding='utf-8')
img_path = conf.get("ctpn", "img_path")
corp_image_path = conf.get("ctpn", "corp_image_path")
gpu = conf.get("ctpn", 'gpu')
checkpoint_path = conf.get("ctpn", "checkpoint_path")

crnn_model_path = conf.get('crnn', "reg_model")


def get_image_path(image_dir):
    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    for parent, dirnames, filenames in os.walk(image_dir):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    print('Find {} images'.format(len(files)))
    return files


def resize_image(img):
    img_size = img.shape
    im_size_min = np.min(img_size[0:2])
    im_size_max = np.max(img_size[0:2])

    im_scale = float(600) / float(im_size_min)
    if np.round(im_scale * im_size_max) > 1200:
        im_scale = float(1200) / float(im_size_max)
    new_h = int(img_size[0] * im_scale)
    new_w = int(img_size[1] * im_scale)

    new_h = new_h if new_h // 16 == 0 else (new_h // 16 + 1) * 16
    new_w = new_w if new_w // 16 == 0 else (new_w // 16 + 1) * 16

    re_im = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return re_im, (new_h / img_size[0], new_w / img_size[1])


def detect(argv=None):
    if os.path.exists(corp_image_path):
        shutil.rmtree(corp_image_path)
    os.makedirs(corp_image_path)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu

    with tf.get_default_graph().as_default():
        input_image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_image')
        input_im_info = tf.placeholder(tf.float32, shape=[None, 3], name='input_im_info')

        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        bbox_pred, cls_pred, cls_prob = model.model(input_image)

        variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
        saver = tf.train.Saver(variable_averages.variables_to_restore())

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            ckpt_state = tf.train.get_checkpoint_state(checkpoint_path)
            model_path = os.path.join(checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
            print('Restore from {}'.format(model_path))
            saver.restore(sess, model_path)

            im_fn = get_image_path(img_path)[0]
            print('===============')
            print(im_fn)
            start = time.time()
            try:
                im = cv2.imread(im_fn)[:, :, ::-1]
            except:
                print("Error reading image {}!".format(im_fn))

            img, (rh, rw) = resize_image(im)
            h, w, c = img.shape
            im_info = np.array([h, w, c]).reshape([1, 3])
            bbox_pred_val, cls_prob_val = sess.run([bbox_pred, cls_prob],
                                                   feed_dict={input_image: [img],
                                                              input_im_info: im_info})

            textsegs, _ = proposal_layer(cls_prob_val, bbox_pred_val, im_info)
            scores = textsegs[:, 0]
            textsegs = textsegs[:, 1:5]

            textdetector = TextDetector(DETECT_MODE='H')
            boxes = textdetector.detect(textsegs, scores[:, np.newaxis], img.shape[:2])
            boxes = np.array(boxes, dtype=np.int)

            cost_time = (time.time() - start)
            print("cost time: {:.2f}s".format(cost_time))

            for i, box in enumerate(boxes):
                left = box[0]
                top = box[1]
                width = box[2] - box[0]
                height = box[7] - box[1]
                crop = img[top:top+height, left:left+width]
                cop_path = os.path.join(corp_image_path, os.path.splitext(os.path.basename(im_fn))[0])+'_'+str(i)+'.jpg'
                cv2.imwrite(cop_path, crop)
                with open(os.path.join(corp_image_path, os.path.splitext(os.path.basename(im_fn))[0]) + '_'+str(i)+".txt",
                          "w") as f:
                    line = ",".join(str(box[k]) for k in range(8))
                    line += "\r\n"
                    f.writelines(line)

            #     cv2.polylines(img, [box[:8].astype(np.int32).reshape((-1, 1, 2))], True, color=(0, 255, 0),
            #                   thickness=2)
            # img = cv2.resize(img, None, None, fx=1.0 / rh, fy=1.0 / rw, interpolation=cv2.INTER_LINEAR)
            # cv2.imwrite(os.path.join(corp_image_path, os.path.basename(im_fn)), img[:, :, ::-1])

            with open(os.path.join(corp_image_path, os.path.splitext(os.path.basename(im_fn))[0]) + ".txt",
                      "w") as f:
                for i, box in enumerate(boxes):
                    line = ",".join(str(box[k]) for k in range(8))
                    line += "," + str(scores[i]) + "\r\n"
                    f.writelines(line)


def recognition():
    alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'

    model = crnn.CRNN(32, 1, 37, 256)
    if torch.cuda.is_available():
        model = model.cuda()
    print('loading pretrained model from %s' % crnn_model_path)
    model.load_state_dict(torch.load(crnn_model_path))

    converter = utils.strLabelConverter(alphabet)

    transformer = dataset.resizeNormalize((100, 32))
    image_paths = get_image_path(img_path)
    for image_path in image_paths:
        image = Image.open(image_path).convert('L')
        image = transformer(image)
        if torch.cuda.is_available():
            image = image.cuda()
        image = image.view(1, *image.size())
        image = Variable(image)

        model.eval()
        preds = model(image)

        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)

        preds_size = Variable(torch.IntTensor([preds.size(0)]))
        raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
        sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
        print('%-20s => %-20s' % (raw_pred, sim_pred))


if __name__ == '__main__':
    tf.app.run(detect)
    recognition()


