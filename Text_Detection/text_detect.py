import os
import time
from collections import OrderedDict

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import Text_Detection.craft_utils as craft_utils
import Text_Detection.file_utils as file_utils
import Text_Detection.imgproc as imgproc
from Text_Detection.craft import CRAFT

result_folder = './Text_Detection/result/'
if not os.path.isdir(result_folder):
    os.mkdir(result_folder)


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def test_net(net, image, text_threshold, link_threshold, low_text, poly, refine_net=None):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, square_size=1280,
                                                                          interpolation=cv2.INTER_LINEAR,
                                                                          mag_ratio=1.5)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)

    x = torch.from_numpy(x).permute(2, 0, 1)  # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))  # [c, h, w] to [b, c, h, w]

    if torch.cuda.is_available():
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0, :, :, 0].cpu().data.numpy()
    score_link = y[0, :, :, 1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0, :, :, 0].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    t1 = time.time() - t1

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    return boxes, polys, ret_score_text


class Detect:
    def __init__(self):
        self.net = CRAFT()

        if torch.cuda.is_available():
            print("device set on cuda")
            self.net.load_state_dict(copyStateDict(torch.load('./Text_Detection/weights/craft_mlt_25k.pth')))
            self.net = self.net.cuda()
            self.net = torch.nn.DataParallel(self.net)
            cudnn.benchmark = False
        else:
            print('device set on cpu')
            self.net.load_state_dict(
                copyStateDict(torch.load('./Text_Detection/weights/craft_mlt_25k.pth', map_location='cpu')))

        self.net.eval()
        self.refine_net = None

    def getDetection(self):
        image_list, _, _ = file_utils.get_files('./Text_Detection/data')
        t = time.time()

        for k, image_path in enumerate(image_list):
            print("Test image {:d}/{:d}: {:s}".format(k + 1, len(image_list), image_path), end='\r')
            image = imgproc.loadImage(image_path)

            bboxes, polys, score_text = test_net(self.net, image, text_threshold=0.7, link_threshold=0.4, low_text=0.4,
                                                 poly=False, refine_net=self.refine_net)

            # save score text
            filename, file_ext = os.path.splitext(os.path.basename(image_path))
            mask_file = result_folder + "/res_" + filename + '_mask.jpg'
            cv2.imwrite(mask_file, score_text)

            file_utils.saveResult(image_path, image[:, :, ::-1], polys, dirname=result_folder)

        print("elapsed time : {}s".format(time.time() - t))


if __name__ == '__main__':
    detect = Detect()
