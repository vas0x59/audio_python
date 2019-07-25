import cv2
name = input()
cap = cv2.VideoCapture(name)
ret = True
i = 0
while ret:
    ret, frame = cap.read()
    i+=1
print(i)
cap.release()