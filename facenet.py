import numpy as np
import sys
import cv2
import os
#import FaceDetection.TestFace as face_recognition
#from FaceDetection.TestFace import face_encodings

from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
from time import localtime
import sys
import pickle as plk
from easydict import EasyDict as edict
from torchvision import transforms as trans
import torch
import string
import random
sys.path.append('F:/GitHub项目/face_align')
import My_Function_lib
import face_preprocess
sys.path.append('F:/GitHub项目/InsightFace_Pytorch/InsightFace_Pytorch-master')
from Learner import face_learner
from utils import load_facebank, draw_box_name, prepare_facebank,load_facebank
fontC = ImageFont.truetype('F:/GitHub项目/Face-recognition-with_GUI/platech.ttf', 16, 0)

def get_config():
    conf = edict()
    conf.use_mobilfacenet = False
    conf.use_lightcnn = False
    conf.drop_ratio = 1.0
    conf.net_mode = 'ir_se'
    conf.net_depth = 50
    conf.facebank_path = 'F:/GitHub项目/Face_recong_test/feature/total_feature'
    conf.model_path = 'F:/GitHub项目/InsightFace_Pytorch/InsightFace_Pytorch-master/work_space'
    conf.input_size = [112, 112]
    conf.test_transform = trans.Compose([
        trans.ToTensor(),
        trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    conf.threshold = 0.8
    conf.face_limit = 5
    conf.min_face_size = 100
    # when inference, at maximum detect 10 faces in one image, my laptop is slow
    return conf

def cv2ImgAddText(img, text, left, top, textColor=(0, 0, 255), textSize=50):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "font/simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text((left, top), text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

class FaceDet(object):
    '''
    人脸检测+识别
    需要识别的脸放到那个faces文件夹里面
    '''
    def __init__(self,register=False):
        self.conf = get_config()
        self.conf.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.mtcnn = My_Function_lib.MTCNN()
        print('mtcnn loaded')

        self.learner = face_learner(self.conf, True)
        if self.conf.device.type == 'cpu':
            self.learner.load_state(self.conf, 'best.pth', True)
        else:
            self.learner.load_state(self.conf, 'final.pth', True)
        self.learner.model.eval()
        print('learner loaded')
        if register==False:
          self.targets, self.names = load_facebank(self.conf)
        print('facebank updated')
        self.process_this_frame = True
        self.Img=None

        # self.input_size = 300  # 输入大小
        # self.process_this_frame = True  # 隔帧检测的bool变量
        # self.base_path = './faces'  # 人脸文件夹
        # self.init_raw_face()  # 初始化操作（加载文件夹中人脸）
        # self.current_name = 'WTF'

    def init_raw_face(self):

        self.frame = None

        self.known_face_encodings = []  # 人脸编码
        self.known_face_names = []  # 人脸姓名

        faces = os.listdir(self.base_path)

        for name in faces:
            tmp = os.path.join(self.base_path, name)
            for pt in os.listdir(tmp):
                print('loading {}……'.format(name))
                self.known_face_names.append(name)

                image = face_recognition.load_image_file(
                    os.path.join(tmp, pt))  # 人脸检测
                encoding = face_encodings(image)[0]  # 人脸编码

                self.known_face_encodings.append(encoding)

        self.process_this_frame = True

    def get_time(self, name):

        out = '\n{}年 {}月 {}日\n时间：{}\n识别身份：{}\n签到成功！'
        t = localtime()
        out = out.format(
            t.tm_year, t.tm_mon, t.tm_mday,
            str(t.tm_hour)+':'+str(t.tm_min),
            name
        )

        return out

    def regist(self,im,text):
        face_save='path'  #Save register face img
        self.Img = im[..., ::-1]
        image = self.Img.copy()
        boxes, points = self.mtcnn.detect(image)
        if len(boxes) > 0:
            boxes, points = self.mtcnn.get_largest_face(boxes, points, image.shape[:2])
            point = (points[:, 0].reshape(2, 5)).T
            face = face_preprocess.preprocess(image, landmark=point)
            save_path=os.path.join(face_save,text)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            cv2.imencode('.png', face[..., ::-1])[1].tofile(os.path.join(
                save_path, '{}.png'.format(text)))
            with torch.no_grad():
                face = Image.fromarray(face)
                embedding=self.learner.model(self.conf.test_transform(face).to(self.conf.device).unsqueeze(0))
                data = {'name': text, 'embedding': embedding}
                save_file = str(self.conf.facebank_path).replace('\\','/') + '/' + text + '.pkl'
                writer = open(save_file, 'wb')
                plk.dump(data, writer)
                writer.close()
                print('regist done!!!')
        else:
            print('detect face err')


    def detect_and_recognition(self, im):
        save_path='F:/GitHub项目/Face_recong_test/save_test_nir'
        if self.process_this_frame:
            self.Img = im[..., ::-1]
            image = self.Img.copy()
            boxes, points = self.mtcnn.detect(image)
            faces = []
            if len(boxes) > 0:
                boxes, points = self.mtcnn.get_largest_face(boxes, points, image.shape[:2])
                for i in range(len(boxes)):
                    point = (points[:, i].reshape(2, 5)).T
                    warped = face_preprocess.preprocess(image, landmark=point)
                    warped = Image.fromarray(warped)
                    faces.append(warped)

                boxes = boxes.astype(int)
                boxes = boxes + [-1, -1, 1, 1]  # personal
                results, score = self.learner.infer(self.conf, faces,self.targets)
                for idx, bbox in enumerate(boxes):
                    im = cv2ImgAddText(im, self.names[results[idx] + 1], bbox[0], bbox[1] - 60, (0, 255, 0), 70)
                    cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 6)
                save_dir=os.path.join(save_path,self.names[results[idx] + 1])
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                img_name = ''.join(random.sample(string.ascii_letters + string.digits, 8)) + '.png'
                cv2.imencode('.png', image[..., ::-1])[1].tofile(os.path.join(
                    save_dir, img_name))


        #self.process_this_frame = not self.process_this_frame
        return im[..., ::-1]

if __name__ == '__main__':

    cap = cv2.VideoCapture('../kun.mp4')
    det = FaceDet()
    video_width = int(cap.get(3))
    video_height = int(cap.get(4))
    fps = int(cap.get(5))
    # fps = 15
    print(fps)
    # fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') #opencv3.0
    # videoWriter = cv2.VideoWriter(
    #     'detected.mp4', fourcc, fps, (542, 300))

    while True:

        _, frame = cap.read()
        if frame is None:
            break
        frame = det.detect_and_recognition(frame)
        cv2.imshow('a', frame)
        # videoWriter.write(im)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    # videoWriter.release()
    cv2.destroyAllWindows()
