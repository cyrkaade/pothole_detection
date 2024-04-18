import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import Tracker
import os

fcount = 0

def imgwrite(img):
    global fcount
    fcount += 1
    filename = '%s.png' % fcount
    cv2.imwrite(os.path.join(r"C://Users//jummd//Downloads//imgs", filename), img)

def videoanalysis(video):
    cap = cv2.VideoCapture(video)
    model = YOLO('best.pt')
    my_file = open("coco.txt", "r")
    data = my_file.read()
    class_list = data.split("\n")
    tracker = Tracker()   
    area = [(0,0), (0,800), (1920,800), (1920,0)]

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps)

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_interval != 0:
            continue

        frame = cv2.resize(frame, (1920, 1080))
        results = model.predict(frame)
        boxes = results[0].boxes.data
        px = pd.DataFrame(boxes).astype("float")
        list = []
        for index, row in px.iterrows():
            x1 = int(row[0])
            y1 = int(row[1])
            x2 = int(row[2])
            y2 = int(row[3])
            d = int(row[5])
            c = class_list[d]
            if 'pothole' in c:
                list.append([x1, y1, x2, y2])

        bbox_idx = tracker.update(list)
        save_frame = False
        for bbox in bbox_idx:
            x3, y3, x4, y4, id = bbox
            results = cv2.pointPolygonTest(np.array(area, np.int32), ((x4, y4)), False)
            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
            if results >= 0:
                save_frame = True

        if save_frame:
            imgwrite(frame)
        
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()

videoanalysis('potholes_test.mp4')
