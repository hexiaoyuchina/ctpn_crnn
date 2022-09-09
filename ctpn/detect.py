# coding=utf-8
import os
import shutil
import sys
import time
import cv2

import numpy as np
import tensorflow as tf

from ctpn.nets import model_train as model
from ctpn.utils.rpn_msr.proposal_layer import proposal_layer
from ctpn.utils.text_connector.detectors import TextDetector

import config
img_path = config.img_path
corp_image_path = config.corp_image_path
gpu = config.gpu
checkpoint_path = config.checkpoint_path
write_line_path = config.write_line_path

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


def get_real_bbox(box, rh, rw):
    real_index = []
    for index in range(0, 8, 2):
        x, y = box[index:index + 2]
        print(x, y)
        x /= rh
        y /= rw
        real_index.append(str(int(x)))
        real_index.append(str(int(y)))
    return ','.join(real_index)

def detect():
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
                    line = get_real_bbox(box, rh, rw)
                    line += "\r\n"
                    f.writelines(line)

                cv2.polylines(img, [box[:8].astype(np.int32).reshape((-1, 1, 2))], True, color=(0, 255, 0),
                              thickness=2)
            img = cv2.resize(img, None, None, fx=1.0 / rh, fy=1.0 / rw, interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(os.path.join(write_line_path, os.path.basename(im_fn)), img[:, :, ::-1])

            with open(os.path.join(write_line_path, os.path.splitext(os.path.basename(im_fn))[0]) + ".txt",
                      "w") as f:
                for i, box in enumerate(boxes):
                    line = get_real_bbox(box, rh, rw)
                    line += "," + str(scores[i]) + "\r\n"
                    f.writelines(line)
