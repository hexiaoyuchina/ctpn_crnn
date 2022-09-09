# coding=utf-8
import os
from ctpn.detect import detect
import torch
from torch.autograd import Variable
from crnn import utils
from crnn import dataset
from PIL import Image
import crnn.models.crnn as crnn
import config

crnn_model_path = config.reg_model
img_path = config.corp_image_path

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
        with open(os.path.join(corp_image_path, os.path.splitext(os.path.basename(image_path))[0]) + "text.txt",
                  "w") as f:
            f.write(sim_pred)



if __name__ == '__main__':
    detect()
    recognition()


