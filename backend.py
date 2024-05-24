from flask import Flask, request, jsonify,render_template
import os
import cv2
import dlib
import numpy as np
import json
from face_module.liveness.liveness_detection import is_blink_3, has_true_and_false
import face_module.faceDetection
import face_module.identification.face_id
import logging as log
import config as cn

app = Flask(__name__)

log.basicConfig(filename='log.txt', level=log.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

current_dir = os.path.dirname(os.path.abspath(__file__))
id_recognition_model_path = os.path.join(current_dir, f"face_module/identification/modelos/id_recognition_model.xml")
landmark_model_path = os.path.join(current_dir, "face_module/liveness/shape_predictor_68_face_landmarks.dat")
labels_path = os.path.join(current_dir, "face_module/identification/labels_ids.json")

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(id_recognition_model_path)

detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor(landmark_model_path)

with open(labels_path, 'r') as file:
    labels = json.load(file)

liveness_images = {}  
image_threshold = 24
app.config['UPLOAD_FOLDER'] = 'uploads'

@app.route('/test',methods=['GET'])
def test():
    response={'Status':40, 'Message':'OK'}
    return jsonify(response)


def upload(video_bytes):
    video_filename = 'temp_file'
    video_file_path = os.path.join(current_dir,app.config['UPLOAD_FOLDER'], video_filename)
     # Write video bytes to a temporary file
    with open(video_file_path,'wb') as file:
        file.write(video_bytes)
    return  video_file_path

@app.route('/identify', methods=['POST'])
def identify():
    try:
        log.info('Iniciando identificação')
        print('iniciando identificacao')
        video_bytes=request.data
        log.info('Salvando Video')
        print('salvando o video')
        video_path=upload(video_bytes)
        video= cv2.VideoCapture(video_path)


        log.info('Obtendo Arquivo')
        # Check if the video opened successfully
        if not video.isOpened():
            print('Error: Unable to open video file')
            return 'Error: Unable to open video file'

        i=0
        #Inicializa lista de resultados
        results = []
        liveness_images = []
        log.info('Iterando video')
        #Itera o video
        while True:
            print(f'imagem numero: {i}')
            ret,frame=video.read()
            
            #Se o video acabou sai do loop
            if not ret:
                print('Video encerrado')
                break

            #Transforma a imagem em cinza        
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
            #Detecta Faces
            faces = detector(gray)
            print(f'{len(faces)} rostos detectados')
            
            #Se tiver mais de uma pessoa retorna erro
            if len (faces)>1:
                return jsonify({'status':'Error: more then one person on the video'})
            if len (faces)==0:
                return jsonify({'status':'Error: No faces Detected'})

            #Se for o primeiro Frame alinha faces
            if i==0:
                faces = face_module.faceDetection.alinha_face(gray, faces)

            #Itera todos os rostos
            for face in faces:
                #Identifica a pessoa no primeiro frame
                if i==0:
                    #inicializa coordenadas da face
                    x = face['coordinates']['x']
                    y = face['coordinates']['y']
                    w = face['coordinates']['w']
                    h = face['coordinates']['h']
                    #Reconhece a face
                    id = face_module.identification.face_id.recognize_face(recognizer, face['image'], label=labels)
                    name = id['name']
                    confidence = id['confidence']
                    print(f'O nome do Usuário é  {name}')
                
                # Liveness detection se não é o primeiro frame
                if i!=0:
                    live = False
                    isBlink = is_blink_3(face, gray, landmark_model_path)
                    liveness_images = append_and_truncate(liveness_images, isBlink, image_threshold)
                    print(f'O frame esta piscando: {isBlink}')
            i=i+1
            
        if len(liveness_images) > 0:
            live = has_true_and_false(liveness_images)
            
        results.append({"name": name, "confidence": confidence, "liveness": live})
        
        video.release() 
        log.info('Identificação Finalizada')
        return jsonify(results)

    except Exception as e:
        log.warning(f'Erro: {e}')
        print(f'erro: {e}')
        response={'Status':'Unknown Error'}
        return jsonify(response)
 

def append_and_truncate(lst, item, size):
    lst.append(item)
    if len(lst) > size:
        del lst[0]
    return lst

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    print('DEV')
    if cn.dev:
        app.run(debug=True)
    else:
        app.run(host='0.0.0.0', port=5000)
    