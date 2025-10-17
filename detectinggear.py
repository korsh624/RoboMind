from ultralytics import YOLO
import cv2
# Загрузка обученной модели
model = YOLO("best.pt")  # путь к вашим весам
savecadr=False
robotdetected=False
colorobots=[]
x_center=0
y_center=0
# Открываем видео
cap = cv2.VideoCapture('input.mp4')


# Проверяем, открылось ли видео
if not cap.isOpened():
    print("Ошибка: Не удалось открыть видео!")
    print(colorobots)
    exit()

# Читаем видео и обрабатываем каждый кадр
while True:
    cropped_img = 0
    ret, frame = cap.read()
    frame=cv2.resize(frame, (640,480))
    # cv2.line(frame, (410, 0), (410, 480), (0, 0, 255), thickness=3)
    # cv2.line(frame, (0, 240), (640, 240), (0, 0, 255), thickness=3)
    if not ret:
        break  # видео закончилось
    results = model.predict(frame, conf=0.1)  # conf - порог уверенности
    for result in results:
        if result.boxes:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # координаты bbox
                cropped_img = frame[y1:y1 + y2, x1:x1 + x2]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                x_center=x1+(x2-x1)//2
                y_center = y1+(y2 - y1) // 2
                cv2.circle(frame, (int(x_center), int(y_center)), 5, (0, 0, 255), cv2.FILLED)
                # cv2.imshow("Robot Detection", cropped_img)
        else:
            robotdetected=False
            colorrobot=None
            savecadr = False
            print("no robot")
    print(*colorobots)
    aim1=(270, 320)
    aim2=(320, 380)
    aim_x_center = aim1[0] + (aim2[0] - aim1[0]) // 2
    aim_y_center = aim1[1] + (aim2[1] - aim1[1]) // 2
    cv2.circle(frame, (int(aim_x_center), int(aim_y_center)), 5, (0, 0, 255), cv2.FILLED)

    cv2.rectangle(frame, aim1,aim2 , (255,0, 0), 2)
    cv2.putText(frame,f'aim_x_center={aim_x_center}aim_y_center={aim_y_center}',(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
    cv2.putText(frame,f'x_center={x_center}y_center={y_center}',(10,60),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)

    cv2.imshow("YOLO Robot Detection frame", frame)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()