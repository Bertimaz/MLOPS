import streamlit as st
import cv2
import numpy as np
import requests
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import json
import config

st.title("Face Recognition FIAP 4DTSR")

option = st.selectbox('Escolha um método:', ('Vídeo', 'Webcam'))

if config.dev:
    # URL de produção
    url = "http://<YOUR-AWS-IP>:5000/identify"
else:
    # Url de desenvolvimento
    url='http://127.0.0.1:5000/identify'

if option == 'Vídeo':
    # Carrega o vidoe na variavel uploaded_file
    uploaded_file = st.file_uploader("Escolha um arquivo de vídeo...", type=["mp4", "avi", "mov"])
    # Se houver video
    if uploaded_file is not None:
        #Apresenta o video na tela
        st.video(uploaded_file)
        #Carrega o video em bytes 
        video_bytes = uploaded_file.read() 
        st.write("Classificando...")

        # Faz o request
        response = requests.post(url, data=video_bytes, verify=False)
        # Escreve na tela a identificacao
        st.write(response.content)

        json_data=response.json()
        json_string=json.dumps(json_data)
        st.write(json_string)

# elif option == 'Use Webcam':
#     st.write("Webcam will start below. Please allow access if prompted.")
#     run = st.checkbox('Start Webcam')

#     FRAME_WINDOW = st.image([])

#     cap = cv2.VideoCapture(0)

#     #Inicializa uma lista vazio para salvar frames encoded como bytes
#     frame_bytes_list = []

#     while run:
#         ret, frame = cap.read()
#         if not ret:
#             st.write("Failed to capture video")
#             break
        
#         frame_bytes_list.append(frame.tobytes())
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         FRAME_WINDOW.image(frame_rgb)

#         _, img_encoded = cv2.imencode('.jpg', frame)
#         files = {'image': ('frame.jpg', img_encoded.tobytes(), 'image/jpeg')}
#         response = requests.post(url, files=files)
#         result = response.json()

#         for res in result:
#             name = res['name']
#             confidence = res['confidence']
#             cv2.putText(frame_rgb, f"{name} - {confidence}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

#         FRAME_WINDOW.image(frame_rgb)

#     cap.release()

elif option == 'Webcam':
    st.write("Webcam will start below. Please allow access if prompted.")
    run = st.checkbox('Start Webcam')

    FRAME_WINDOW = st.image([])

    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to capture video")
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame_rgb)

        _, img_encoded = cv2.imencode('.jpg', frame)
        files = {'image': ('frame.jpg', img_encoded.tobytes(), 'image/jpeg')}
        response = requests.post(url, files=files)
        result = response.json()

        for res in result:
            name = res['name']
            confidence = res['confidence']
            cv2.putText(frame_rgb, f"{name} - {confidence}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        FRAME_WINDOW.image(frame_rgb)

    cap.release()