import numpy as np
import cv2
import pickle


face_cascade=cv2.CascadeClassifier("/home/george/PycharmProjects/OpenCamera/haarcascade_frontalface_alt2.xml")
eye_cascade=cv2.CascadeClassifier("/media/george/Personal/Python Projects/frontalEyes/frontalEyes.xml")

recognizer=cv2.face.LBPHFaceRecognizer_create()
recognizer.read('recognizer/trainingData.yml')

labels={"person_name": 1}
with open("/home/george/PycharmProjects/OpenCamera/labels.pickle", 'rb') as f:
    og_labels=pickle.load(f)
    labels={v:k for k,v in og_labels.items()}

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, img = cap.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=4)

    for(x,y,w,h) in faces:
        print(x,y,w,h)
        roi_gray=gray[y:y+h,x:x+w]  #(ycord_start,ycord2_end)
        roi_color = img[y:y + h, x:x + w]

        #recognizer# deep learned model predict keras tensorflow pytorch
        ID, conf = recognizer.predict(roi_gray)
        if conf>=45 and conf<= 85:
            print(ID)
            print(labels[ID])
            font=cv2.FONT_HERSHEY_SIMPLEX
            name=labels[ID]
            color=(255,255,255)
            stroke=2
            cv2.putText(img,name,(x,y),font,1,color,stroke,cv2.LINE_AA)


        # img_item="images/George/image9.png"
        # img_item1="images/George/image10.png"
        #
        # cv2.imwrite(img_item,roi_gray)
        # cv2.imwrite(img_item1,roi_color)

        color=(255,0,0) #bgr 0-255
        stroke=2
        width=x + w
        height=y+h
        cv2.rectangle(img, (x,y), (width,height),color,stroke)

        eyes = eye_cascade.detectMultiScale(roi_gray)

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('frame',img)
    if cv2.waitKey(20) & 0xFF == ord("q"):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()