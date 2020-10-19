import utils.gpu as gpu
from model.yolov3 import Yolov3
from utils.visualize import *

from eval.evaluator import Evaluator

import config.yolov3_config_voc as cfg

import os
import argparse
import torch
import cv2


class Tester(object):
    def __init__(self,
                 weight_path=None,
                 gpu_id=0,
                 img_size=544,
                 test_dir=None,
                 eval=False
                 ):
        self.img_size = img_size
        self.__num_class = cfg.DATA["NUM"]
        self.__conf_threshold = cfg.TEST["CONF_THRESH"]
        self.__nms_threshold = cfg.TEST["NMS_THRESH"]
        self.__device = gpu.select_device(gpu_id, force_cpu=True)
        self.__multi_scale_test = cfg.TEST["MULTI_SCALE_TEST"]
        self.__flip_test = cfg.TEST["FLIP_TEST"]

        self.__test_dir = test_dir
        self.__eval = eval
        self.__classes = cfg.DATA["CLASSES"]

        self.__model = Yolov3().to(self.__device)

        self.__load_model_weights(weight_path)

        self.__evalter = Evaluator(self.__model, visiual=False)


    def __load_model_weights(self, weight_path):
        print("loading weight file from : {}".format(weight_path))

        weight = os.path.join(weight_path)
        chkpt = torch.load(weight, map_location=self.__device)
        self.__model.load_state_dict(chkpt)
        print("loading weight file is done")
        del chkpt


    def test(self):
        if self.__test_dir:
            if os.path.isfile(self.__test_dir):
                if self.__test_dir.endswith('.jpg'):
                    imgs = [os.path.split(self.__test_dir)[-1]]
                    self.__test_dir = os.path.split(self.__test_dir)[0]
            elif os.path.isdir(self.__test_dir):
                imgs = [f for f in os.listdir(self.__test_dir) if f.endswith('.jpg')]
            
            for v in imgs:
                path = os.path.join(self.__test_dir, v)
                print("test images : {}".format(path))

                img = cv2.imread(path)
                assert img is not None

                # bboxes_prd = self.__evalter.get_bbox(img)
                bboxes_prd = self.__evalter.get_bbox_cleaned(img)
                if bboxes_prd.shape[0] != 0:
                    boxes = bboxes_prd[..., :4]
                    class_inds = bboxes_prd[..., 5].astype(np.int32)
                    scores = bboxes_prd[..., 4]

                    visualize_boxes(image=img, boxes=boxes, labels=class_inds, probs=scores, class_labels=self.__classes)
                    path = os.path.join(cfg.PROJECT_PATH, "output/{}".format(v))

                    cv2.imwrite(path, img)
                    print("saved images : {}".format(path))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_path', type=str, default='weight/best.pt', help='weight file path')
    parser.add_argument('--test_dir', type=str, default='./test', help='test data path or None')
    parser.add_argument('--eval', action='store_true', default=False, help='eval the mAP or not')
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    opt = parser.parse_args()

    predictor = Tester( weight_path=opt.weight_path,
                        gpu_id=opt.gpu_id,
                        eval=opt.eval,
                        test_dir=opt.test_dir)

    predictor.test()
