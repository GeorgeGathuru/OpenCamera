import cv2
import os
import pickle
import numpy as np
from PIL import Image, ImageEnhance

BASE_DIR= os.path.dirname(os.path.abspath(__file__))
image_dir= os.path.join(BASE_DIR,"images")

face_cascade=cv2.CascadeClassifier("/media/george/Personal/Python Projects/frontalFace10/haarcascade_frontalface_alt2.xml")

recognizer=cv2.face.LBPHFaceRecognizer_create()

current_id=0
label_ids={}
y_labels=[]
x_train=[]


for root, dirs, files in os.walk(image_dir):
     for file in files:
         if file.endswith("png") or file.endswith("jpg"):
             path=os.path.join(root, file)
             label=os.path.basename(os.path.basename(path)).replace(" ", "-").lower()
             # print(label,path)
             y_labels.append(label) #some number
             x_train.append(path) #verify this image, turn into nummpy array , Graysscale

             pil_image = Image.open(path).convert("L")
             size=(500,500)
             final_image=pil_image.resize(size, Image.ANTIALIAS)
             image_array=np.array(final_image,'uint8')
             x_train.append(image_array)


             # print(image_array)

             if not label in label_ids:
                 label_ids[label]=current_id
                 current_id +=1
             id_=label_ids[label]

             y_labels.append(label_ids)

             # print(label_ids)

             faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=4)

             for(x,y,w,h) in faces:
                 roi=image_array[y:y+h,x:x+w]
                 # x_train.append(roi)
                 y_labels.append(id_)



print(x_train )
print(y_labels)

with open("/home/george/PycharmProjects/OpenCamera/labels.pickle", 'wb') as f:
	pickle.dump(label_ids, f)

recognizer.train(np.array(x_train),np.array(y_labels))
recognizer.save("/home/george/PycharmProjects/OpenCamera/trainner.yml")















