import cv2
from time import time

import face_module.faceDetection
import face_module.identification
import face_module.identification.face_id
import face_module.liveness.liveness_detection
from face_module.liveness.liveness_detection import is_blink,is_blink_2, is_blink_3
import dlib
import os

cam = cv2.VideoCapture(0) 
cam.set(cv2.CAP_PROP_FPS, 30)


def append_and_truncate(lst, item,size):
    lst.append(item)
    if len(lst) > size:
        del lst[0]
    return lst



# Pegando modelo de identificação
# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
id_recognition_model_path=os.path.join(current_dir,r"face_module\identification\modelos\id_recognition_model.xml")
# print(id_recognition_model_path)
recognizer=face_module.identification.face_id.get_recognizer(id_recognition_model_path)

##Obtendo modelo d lib
landmark_model_path=os.path.join(current_dir,r'face_module\liveness\shape_predictor_68_face_landmarks.dat')

#pegando labels de usuarios
labels_path=os.path.join(current_dir,'face_module\identification\labels_ids.json')
labels=face_module.identification.face_id.load_labels(labels_path)

# Lista de imagens para trakear piscadas
liveness_images={}  ###{'usuario 1':[True,False....]}             #{'usuario 1':[img01,img02....]}

#   Numero de imagens salvas
image_treshold=48



# Initializing the Models for Landmark and  
# face Detection 
detector = dlib.get_frontal_face_detector() 
landmark_predict = dlib.shape_predictor(landmark_model_path) 


count=0

while True:
    # Ler o frame da camera
    #Se o video finalizar recomeçar
    if cam.get(cv2.CAP_PROP_POS_FRAMES) == cam.get( 
            cv2.CAP_PROP_FRAME_COUNT): 
        cam.set(cv2.CAP_PROP_POS_FRAMES, 0) 
    else:

        ret, frame = cam.read()

        
        #Prepara a imagem
        ### print(imagem.shape)
        gray= cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # print(f'Type= {type(gray)}\n\n')
                # detecting the faces 

        faces = detector(gray)
        

        faces=face_module.faceDetection.alinha_face(gray,faces)


        #Para cada uma das faces
        for face in faces:
            # faces.append({'image':cropped, 'coordinates':{'x':x,'y':y,'w':w,'h':h}})
            x=face['coordinates']['x']
            y=face['coordinates']['y']
            w=face['coordinates']['w']
            h=face['coordinates']['h']
            #Desenha um frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            #identifica pessoas
            id=face_module.identification.face_id.recognize_face(recognizer,face['image'],label=labels)
            name=id['name']
            confidence=id['confidence']

            #Liveness
            live=False
            if name !="Unknown":

                if name not in liveness_images:
                    liveness_images[name]=[]
        
                # print(face)
                # isBlink=is_blink_2(face,gray,landmark_model_path)
                isBlink=is_blink_2(frame,landmark_model_path)
                
                isBlink=is_blink_3(face['face'],gray,landmark_model_path)

                liveness_images[name]=append_and_truncate(liveness_images[name],isBlink,image_treshold)
                # print(f'blinks ={liveness_images}')

                if len(liveness_images[name])>0:
                    live=face_module.liveness.liveness_detection.has_true_and_false(liveness_images[name])

       
                # print(f'isblink_3 = {isBlink} /n ')
                # print(liveness_images)

            cv2.putText(frame, f"{name} - Confidence: {round(confidence,0)} Live: {live}", (x-int(w*0.5), y-10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 200, 0), 2) 

        
        # Display the frame with face detection
        cv2.imshow('Capture Faces', frame)

        # Break the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


