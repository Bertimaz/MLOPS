import os
dev=False


if dev:
    # Caminhos de modelos
    current_dir = os.path.dirname(os.path.abspath(__file__))
    id_recognition_model_path = os.path.join(current_dir, f"face_module\identification\modelos\id_recognition_model.xml")
    landmark_model_path = os.path.join(current_dir, "face_module\liveness\shape_predictor_68_face_landmarks.dat")
    classificador_68_path = landmark_model_path