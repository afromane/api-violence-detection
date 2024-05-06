import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
import dlib
from django.conf import settings

class FaceRecognition:
    def __init__(self):
        self.detector = MTCNN()
        shape_predictor_path = settings.BASE_DIR+"/static/dlib/shape_predictor_68_face_landmarks.dat"

        self.predictor = dlib.shape_predictor(shape_predictor_path)

    def detect_and_extract_facial_landmarks(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detect_faces(image)
        facial_landmarks_list = []

        for face in faces:
            if 'box' in face:
                x, y, width, height = face['box']
                if width > 0 and height > 0:
                    margin = 20
                    face_img = image[max(0, y-margin):min(y+height+margin, image.shape[0]),
                                     max(0, x-margin):min(x+width+margin, image.shape[1])]
                    landmarks = self.predictor(gray, dlib.rectangle(x, y, x + width, y + height))
                    facial_landmarks = [(landmark.x, landmark.y) for landmark in landmarks.parts()]
                    facial_landmarks_list.append(facial_landmarks)

                    for (x, y) in facial_landmarks:
                        cv2.circle(image, (x, y), 1, (0, 255, 0), -1)

        return facial_landmarks_list,image
    def extract_faces(img):
        # Créer le détecteur de visage
        detector = MTCNN()

        # Détecter les visages dans l'image
        faces = detector.detect_faces(img)

        extracted_features = []

        # Parcourir les visages détectés et extraire les caractéristiques faciales
        for face in faces:
            # Vérifier si la boîte englobante est valide
            if 'box' in face:
                # Extraire le visage de l'image
                x, y, width, height = face['box']

                # Vérifier si la taille de la région du visage est valide
                if width > 0 and height > 0:
                    margin = 20  # Définir la taille de la marge

                    # Extraire le visage de l'image avec une marge de 20 pixels
                    face_img = img[max(0, y-margin):min(y+height+margin, img.shape[0]),
                                max(0, x-margin):min(x+width+margin, img.shape[1])]

                    facial_landmarks = detect_and_extract_facial_landmarks(face_img)
                    # Ajouter les caractéristiques faciales à la liste
                    extracted_features.append(facial_landmarks)

        return extracted_features
    def detect_faces_and_draw_rectangles(self,image):
        # Créer l'objet détecteur de visage
        detector = MTCNN()
        # Détecter les visages dans l'image
        faces = detector.detect_faces(image)
        # Dessiner des rectangles verts autour des visages détectés
        for face in faces:
            # Extraire les coordonnées du rectangle englobant le visage
            x, y, width, height = face['box']
            # Dessiner un rectangle vert autour du visage détecté
            cv2.rectangle(image, (x, y), (x+width, y+height), (0, 255, 0), 2)

        return image

    def calculate_face_metrics(self, facial_landmarks):
        metrics = {}
        face_width = facial_landmarks[16][0] - facial_landmarks[0][0]
        face_height = facial_landmarks[8][1] - facial_landmarks[27][1]
        eye_distance = facial_landmarks[45][0] - facial_landmarks[36][0]
        left_eye_width = facial_landmarks[39][0] - facial_landmarks[36][0]
        right_eye_width = facial_landmarks[45][0] - facial_landmarks[42][0]
        nose_to_eyes_distance = np.abs(facial_landmarks[27][1] - (facial_landmarks[39][1] + facial_landmarks[42][1]) / 2)
        sinus = nose_to_eyes_distance / eye_distance
        mouth_width = facial_landmarks[54][0] - facial_landmarks[48][0]

        metrics['face_width'] = face_width
        metrics['face_height'] = face_height
        metrics['eye_distance'] = eye_distance
        metrics['left_eye_width'] = left_eye_width
        metrics['right_eye_width'] = right_eye_width
        metrics['nose_to_eyes_distance'] = nose_to_eyes_distance
        metrics['sinus'] = sinus
        metrics['mouth_width'] = mouth_width

        return metrics
""" 
# Utilisation de la classe
img_path = "path/to/image.jpg"
shape_predictor_path = "path/to/shape_predictor_68_face_landmarks.dat"

recognizer = FaceRecognition(shape_predictor_path)
image = cv2.imread(img_path)
facial_landmarks = recognizer.detect_and_extract_facial_landmarks(image)
if facial_landmarks:
    metrics = recognizer.calculate_face_metrics(facial_landmarks[0])
    print("Mesures du visage :", metrics)
 """