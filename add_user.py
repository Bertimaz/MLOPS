import os
import cv2
from tqdm import tqdm
import re
from face_module import faceDetection
import numpy as np
import shutil
import dlib

current_dir = os.path.dirname(os.path.abspath(__file__))
faces_file_folder_path=os.path.join(current_dir,r"face_module\identification\files\caltech_faces")
temp_folder=f'{faces_file_folder_path}/temp'
labels_path=os.path.join(current_dir,r"face_module\identification\labels_ids.json")
id_recognition_model_path=os.path.join(current_dir,r"face_module\identification\modelos\id_recognition_model.xml")
n_pictures=50

# Function to capture images and store in dataset folder
def capture_images(User):
    #Get current dir
   

    # Create a directory to store the captured images
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)

 
    # Open the camera
    cap = cv2.VideoCapture(0)
 
    # Set the image counter as 0
    count = 0
 
    while True:
        # Read a frame from the camera
        ret, frame = cap.read()
 
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
        # Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
 
        # Draw rectangles around the faces and store the images
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{n_pictures- count} fotos faltantes", (x-int(w*0.5), y-10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 200, 0), 2) 

 
            # Store the captured face images in the Faces folder
            cv2.imwrite(f'{temp_folder}/{User} ({count}).jpg', frame)
 
            count += 1
 
        # Display the frame with face detection
        cv2.imshow('Capture Faces', frame)
 
        # Break the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
 
        # Break the loop after capturing a certain number of images
        if count >= n_pictures:
            break
 
    # Release the camera and close windows
    cap.release()
    cv2.destroyAllWindows()

def update_model(new_user_name):
    faces = []
    labels = []
   

    detector = dlib.get_frontal_face_detector() 

    # Load the images from the 'Faces' folder
    for file_name in tqdm(os.listdir(temp_folder)):
        if file_name.endswith('.jpg'):
            # Extract the label (person's name) from the file name
            name = re.split(r'\s+\(.*\)', file_name)[0]
        
            if name==new_user_name:
                # Read the image
                image = cv2.imread(os.path.join(temp_folder, file_name))

                # Convert the frame to grayscale
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
                # Detect faces in the grayscale frame
                dets=detector(gray)
                # Crop the detected face region
                cropped_faces = faceDetection.alinha_face(gray,dets,return_face=True)
                # Se existir apenas uma face, salva a imagem e o label
                if len(cropped_faces)==1:
                    # Pega as informações da imagem na lista
                    face_crop=cropped_faces[0]
                    # Pega a imagem no dicionario
                    face_crop=face_crop['image']
                    # Append the face sample and label to the lists
                    faces.append(face_crop)
                    labels.append(label[name])
                else:
                    print('Imagem com mais de uma face. Rejeitada') 
            else:
                print('Imagem com nome em formato incorreto')

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(id_recognition_model_path)
    recognizer.update(faces,np.array(labels))
    recognizer.save(id_recognition_model_path)



def save_labels(labels):
    import json
    # Specify the file path
    file_path = labels_path

    # Save dictionary to a file
    with open(file_path, 'w') as file:
        json.dump(labels, file)


def load_labels():

    import json

    # Specify the file path
    file_path = labels_path

    # Load dictionary from file
    with open(file_path, 'r') as file:
        loaded_dict = json.load(file)

    return loaded_dict


def move_images(new_user_name):
    print(f'movendo Imagens para {faces_file_folder_path}')
    # Load the images from the 'Faces' folder
    for file_name in tqdm(os.listdir(temp_folder)):
        if file_name.endswith('.jpg'):
            # Extract the label (person's name) from the file name
            name = re.split(r'\s+\(.*\)', file_name)[0]
            if name==new_user_name:
                shutil.move(os.path.join(current_dir,temp_folder,file_name),faces_file_folder_path)
            


# Create a face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
 
# Generate a face recognition model
recognizer = cv2.face.LBPHFaceRecognizer_create()
name=input('Qual Usuário a ser adicionado?')


label=load_labels() 

if name not in label:
    last_label_number=max(label.values())
    label[name]=last_label_number+1



capture_images(name)
update_model(name)
move_images(name)

if name not in label:
    save_labels(label)
