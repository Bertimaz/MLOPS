import cv2

def save_labels(labels):
    import json
    # Specify the file path
    file_path = 'labels_ids.json'

    # Save dictionary to a file
    with open(file_path, 'w') as file:
        json.dump(labels, file)


def load_labels(file_path='labels_ids.json'):

    import json

    # Load dictionary from file
    with open(file_path, 'r') as file:
        loaded_dict = json.load(file)

    return loaded_dict

def get_recognizer(path):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(path)
    return recognizer


def recognize_face(recognizer,imagem,label):
    """Função que reconhece a face de uma imagem cropada"""
 
    # Reverse keys and values in the dictionary
    label_name = {value: key for key, value in label.items()}
    #Preparar Imagem
    # imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    label,confidence=recognizer.predict(imagem)
    if confidence>50:
        return {'name':label_name[label],'confidence':confidence}
    else:
        return {'name':'Unknown','confidence':1-confidence}

dir_path='files/caltech_faces/test'

