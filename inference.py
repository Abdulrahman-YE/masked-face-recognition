import argparse
from email import parser
import os
from pathlib import Path
import shutil
import cv2
from matplotlib.pyplot import gray
import torch
from models import Darknet, load_darknet_weights
from utils import torch_utils
from utils.datasets import LoadImages
from utils.utils import check_file, load_classes, non_max_suppression, plot_one_box, scale_coords
CGF = 'cfg/yolo-fastest.cfg'
WEIGHTS = 'weights/best.weights'
NAMES = 'data/face_mask.names'
AGUMENTED = True
# object confidence threshold
CONF_THRESH = 0.3
# IOU threshold for NMS
IOU_THRESH = 0.6
# class-agnostic NMS
AGNOSTIC_NMS = True
# filter by class
CLASSES = None
NAMES = load_classes('data/face_mask.names')
# Color for binary classification in B,G,R
COLORS = [[0, 69, 255], [170, 178, 32]]

DEVICE = torch_utils.select_device('')


def inference(_img, img_size=512):

    img_size = img_size


    # Init model
    model = Darknet(CGF, img_size)
    # Load weights
    load_darknet_weights(model, WEIGHTS)
    # Eval model
    model.to(DEVICE).eval()


    # Get names and colors
    # Color for binary classification in B,G,R
    colors = [[0, 69, 255], [170, 178, 32]]
    img = torch.zeros((1, 3, img_size, img_size), device=DEVICE)  # init img
    _ = model(img.float()
              ) if DEVICE.type != 'cpu' else None    # run once


    # Inference
    t1 = torch_utils.time_synchronized()
    pred = model(_img, augment=AGUMENTED)[0]
    t2 = torch_utils.time_synchronized()

    # Apply NMS
    pred = non_max_suppression(pred, CONF_THRESH, IOU_THRESH,
                                multi_label=False, classes=CLASSES, agnostic=AGNOSTIC_NMS)

    return pred



def detect(source):
    dataset = LoadImages(source)
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(DEVICE)
        img = img.float()  # uint9 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
                img = img.unsqueeze(0)
        pred = inference(_img=img)
        for i, det in enumerate(pred):
            p, s, im0 = path, '', im0s
            save_path = str(Path(out) / Path(p).name)
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from imgsz to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    # Detection per class
                    n = (det[:, -1] == c).sum()
                    s += '%g %ss, ' % (n, NAMES[int(c)])  # add to string

                # Write result:
                for *xyxy, conf, cls in det:
                    if save_image or view_image:
                        label = '%s %.2f' % (NAMES[int(cls)], conf)
                        print(
                            f'c1({xyxy[0]}, {xyxy[1]}... c2({xyxy[2]}, {xyxy[3]}')
                            
                        resized = im0[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
                        cv2.imwrite(save_path.replace('output', 'pre-process'),resized)

                        plot_one_box(xyxy, im0, label=label, color=COLORS[int(cls)])

                        
            # Print time (inference + NMS)
            print('%sDone' % (s))
            # Save results (image with detections)
            if save_image:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                    vid_writer.write(im0)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--images-path', type=str, default="images", help='Imgaes path')

    opt = parser.parse_args()
    opt.images_path = check_file(opt.images_path)
    save_image = True
    view_image = False
    source = opt.images_path

    out = "output"
    res = "resized"

        

    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    if os.path.exists(res):
        shutil.rmtree(res)  # delete output folder
    os.makedirs(res)  # make new output folder

    with torch.no_grad():
        detect(source)
    if save_image:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        


