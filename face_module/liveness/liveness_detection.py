# Importing the required dependencies 
import cv2  # for video rendering 
import dlib  # for face and landmark detection 
import imutils 
# for calculating dist b/w the eye landmarks 
from scipy.spatial import distance as dist 
# to get the landmark ids of the left and right eyes 
# you can do this manually too 
from imutils import face_utils 
import numpy as np
  
# from imutils import 
  
def is_live(frames: list,n_frames:int,model_path:str):
    """parametros
        frames n_frames ultimos frames em imagem de cinza cropado no rosto
    """
    if len (frames)<n_frames:
        return False
    
    blinks=[]
    for frame in frames:
        blinks.append(is_blink(frame,model_path))

    if has_true_and_false(blinks):
        True
    else:
        False
  

def is_blink(gray,model_path:str):
    """
    Check if a image has blink
    Parameters
    frame: cropped image of a person in gray
    """

    

    # Define the desired width
    desired_width = 640

    # Calculate the aspect ratio
    height, width = gray.shape[:2]
    aspect_ratio = width / height

    # Calculate the new height based on the desired width and aspect ratio
    desired_height = int(desired_width / aspect_ratio)

    # Resize the image
    gray = cv2.resize(gray, (desired_width, desired_height))

    # print(f'is blikn type frame = {type(gray)}')

    # Eye landmarks 
    (L_start, L_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"] 
    (R_start, R_end) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye'] 

    # Variables 
    blink_thresh = 0.45



    # face Detection 
    detector = dlib.get_frontal_face_detector() 
    landmark_predict = dlib.shape_predictor(model_path) 
    faces = detector(gray) 

    for face in faces: 
        # landmark detection 
        shape = landmark_predict(gray, face) 


        # converting the shape class directly 
        # to a list of (x,y) coordinates 
        shape = face_utils.shape_to_np(shape) 

        # parsing the landmarks list to extract 
        # lefteye and righteye landmarks--# 
        lefteye = shape[L_start: L_end] 
        righteye = shape[R_start:R_end] 

    # Calculate the EAR 
        left_EAR = calculate_EAR(lefteye) 
        right_EAR = calculate_EAR(righteye) 


        # Avg of left and right eye EAR 
        avg = (left_EAR+right_EAR)/2

        
        print(avg)
        print(blink_thresh)

        if avg < blink_thresh: 
            return True
        else: 
            return False

        return 'falhou'
       

def has_true_and_false(lst):
    return any(lst) and any(not item for item in lst)


# defining a function to calculate the EAR 
def calculate_EAR(eye): 
  
    # calculate the vertical distances 
    y1 = dist.euclidean(eye[1], eye[5]) 
    y2 = dist.euclidean(eye[2], eye[4]) 
  
    # calculate the horizontal distance 
    x1 = dist.euclidean(eye[0], eye[3]) 
  
    # calculate the EAR 
    EAR = (y1+y2) / x1 
    return EAR 

def is_blink_2(frame,landmark_model_path):
          
    # defining a function to calculate the EAR 
    def calculate_EAR(eye): 
    
        # calculate the vertical distances 
        y1 = dist.euclidean(eye[1], eye[5]) 
        y2 = dist.euclidean(eye[2], eye[4]) 
    
        # calculate the horizontal distance 
        x1 = dist.euclidean(eye[0], eye[3]) 
    
        # calculate the EAR 
        EAR = (y1+y2) / x1 
        return EAR 
        
    detector = dlib.get_frontal_face_detector() 
    landmark_predict = dlib.shape_predictor(landmark_model_path) 

    # Variables 
    blink_thresh = 0.45
    succ_frame = 2
    count_frame = 0
  
    # Eye landmarks 
    (L_start, L_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"] 
    (R_start, R_end) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye'] 
    # converting frame to gray scale to 
    # pass to detector 
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 

    # detecting the faces 
    faces = detector(img_gray) 
    for face in faces: 

        # landmark detection 
        shape = landmark_predict(img_gray, face) 

        # converting the shape class directly 
        # to a list of (x,y) coordinates 
        shape = face_utils.shape_to_np(shape) 

        # parsing the landmarks list to extract 
        # lefteye and righteye landmarks--# 
        lefteye = shape[L_start: L_end] 
        righteye = shape[R_start:R_end] 

        # Calculate the EAR 
        left_EAR = calculate_EAR(lefteye) 
        right_EAR = calculate_EAR(righteye) 

        # Avg of left and right eye EAR 
        avg = (left_EAR+right_EAR)/2
    
    if len(faces)>0:
        return avg
    return -1


def is_blink_3(face,img_gray,landmark_model_path):
          
    # defining a function to calculate the EAR 
    def calculate_EAR(eye): 
    
        # calculate the vertical distances 
        y1 = dist.euclidean(eye[1], eye[5]) 
        y2 = dist.euclidean(eye[2], eye[4]) 
    
        # calculate the horizontal distance 
        x1 = dist.euclidean(eye[0], eye[3]) 
    
        # calculate the EAR 
        EAR = (y1+y2) / x1 
        return EAR 
        
    detector = dlib.get_frontal_face_detector() 
    landmark_predict = dlib.shape_predictor(landmark_model_path) 

    # Variables 
    blink_thresh = 0.45
    succ_frame = 2

  
    # Eye landmarks 
    (L_start, L_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"] 
    (R_start, R_end) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye'] 
    # converting frame to gray scale to 
    # pass to detector 


    # landmark detection 
    shape = landmark_predict(img_gray, face) 

    # converting the shape class directly 
    # to a list of (x,y) coordinates 
    shape = face_utils.shape_to_np(shape) 

    # parsing the landmarks list to extract 
    # lefteye and righteye landmarks--# 
    lefteye = shape[L_start: L_end] 
    righteye = shape[R_start:R_end] 

    # Calculate the EAR 
    left_EAR = calculate_EAR(lefteye) 
    right_EAR = calculate_EAR(righteye) 

    # Avg of left and right eye EAR 
    avg = (left_EAR+right_EAR)/2

    if avg < blink_thresh: 
            return True
    else: 
        return False





if __name__=='__main__':



    landmark_model_path= r'C:\Users\alber\Documents\Projetos - Dados\computer_vision\face_module\liveness\shape_predictor_68_face_landmarks.dat'    

    blink_image=cv2.imread(r'C:\Users\alber\Documents\Projetos - Dados\computer_vision\face_module\liveness\files\fechado.jpg')

    not_blink_image=cv2.imread(r'C:\Users\alber\Documents\Projetos - Dados\computer_vision\face_module\liveness\files\aberto.jpg')
    print(is_blink_2(blink_image,landmark_model_path))
    print(is_blink_2(not_blink_image,landmark_model_path))
        
