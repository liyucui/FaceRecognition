#coding:utf-8
from train_models.mtcnn_model import R_Net
from train_models.train import train
import os


def train_RNet(base_dir, prefix, end_epoch, display, lr):
    """
    train PNet
    :param dataset_dir: tfrecord path
    :param prefix:
    :param end_epoch:
    :param display:
    :param lr:
    :return:
    """
    net_factory = R_Net
    train(net_factory, prefix, end_epoch, base_dir, display=display, base_lr=lr)

if __name__ == '__main__':
    base_dir = '/home/lucy/PycharmProjects/MTCNN-Tensorflow-master/RGB-D-face/imglists/RNet'

    model_name = 'MTCNN'
    model_path = '/home/lucy/PycharmProjects/mtcnn/RGB-D-face/%s_model/RNet_landmark/RNet' % model_name
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    prefix = model_path
    end_epoch = 30
    display = 100
    lr = 0.0005
    train_RNet(base_dir, prefix, end_epoch, display, lr)