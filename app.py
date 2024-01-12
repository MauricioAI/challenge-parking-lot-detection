from pathlib import Path

import cv2
import torch
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import plot_one_box

from flask import Flask, request

app = Flask(__name__)



app.route('/')
def show_bounding_boxes():
    data = request.get_json()
    half = data['device'].type != 'cpu'

    # Load model
    model = attempt_load(data['weights'], map_location=data['device'])  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(data['img_size'], s=stride)  # check img_size

    if half:
        model.half()  # to FP16

    dataset = LoadImages(data['source'], img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if data['device'].type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(data['device']).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(data['device'])
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if data['device'].type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img)[0]

        # Inference
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img)[0]

        # Apply NMS
        pred = non_max_suppression(pred)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    label = f'{names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

            # Stream results
            cv2.imshow(str(p), im0)
            cv2.waitKey(1)  # 1 millisecond
            return f'{s}\n'


if __name__=="__main__":
    app.run(debug=True)