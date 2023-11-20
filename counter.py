from ultralytics import YOLO
import cv2
import cvzone
import pandas as pd
import numpy as np
from vidgear.gears import CamGear
from tracker import *

model = YOLO('models/yolov8s.pt')

stream = CamGear(source='videos/trucks.mp4', logging=True).start()  # YouTube Video URL as input


def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)


cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

my_file = open("models/coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")
# print(class_list)
count = 0
tracker = Tracker()
area2 = [(77,285), (577,350) , (583,321),(88,257)]
area1 = [(8,325), (561,429), (576,378), (35,297)]
entercar = {}
exitcar = {}
entercar_counter = []
exitcar_counter = []
while True:
    frame = stream.read()
    count += 1
    if count % 2 != 0:
        continue

    frame = cv2.resize(frame, (1020, 500))

    results = model.predict(frame)
    #   print(results)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")
    #    print(px)
    list = []
    for index, row in px.iterrows():


        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        if 'car' in c:
            list.append([x1, y1, x2, y2])
        if 'bus' in c:
            list.append(([x1, y1, x2, y2]))
        if 'truck' in c:
            list.append(([x1, y1, x2, y2]))
    bbox_idx = tracker.update(list)
    for bbox in bbox_idx:
        x3, y3, x4, y4, id1 = bbox
        cx = int(x3 + x4) // 2
        cy = int(y3 + y4) // 2
        result = cv2.pointPolygonTest(np.array(area1, np.int32), ((cx, cy)), False)
        if result >= 0:
            entercar[id1] = (cx, cy)
        if id1 in entercar:
            result1 = cv2.pointPolygonTest(np.array(area2, np.int32), ((cx, cy)), False)
            if result1 >= 0:
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 255, 255), 2)
                cvzone.putTextRect(frame, f'{id1}', (x3, y3), 1, 1)
                if entercar_counter.count(id1) == 0:
                    entercar_counter.append(id1)

        ################################FOR BLUE POLYLİNES####################################
        result2 = cv2.pointPolygonTest(np.array(area2, np.int32), ((cx, cy)), False)
        if result2 >= 0:
            exitcar[id1] = (cx, cy)
        if id1 in exitcar:
            result3 = cv2.pointPolygonTest(np.array(area1, np.int32), ((cx, cy)), False)
            if result3 >= 0:
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 255, 255), 2)
                cvzone.putTextRect(frame, f'{id1}', (x3, y3), 1, 1)
                if exitcar_counter.count(id1) == 0:
                    exitcar_counter.append(id1)

    cv2.polylines(frame, [np.array(area1, np.int32)], True, (0, 255, 0), 2)
    cv2.polylines(frame, [np.array(area2, np.int32)], True, (255, 255, 0), 2)
    card = (len(entercar_counter))
    caru = (len(exitcar_counter))
    # cvzone.putTextRect(frame, f'Giris Yapan Araç Sayisi:{card}', (50, 60), 1, 1)
    # cvzone.putTextRect(frame, f'Çıkış Yapan Araç Sayisi:{caru}', (846, 59), 1, 1)
    cvzone.putTextRect(frame, f'Toplam Arac:{caru - card}', (465, 29), 1, 1)
    print(caru-card)

    cv2.imshow("RGB", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break
stream.release()
cv2.destroyAllWindows()

