from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
import os
from django.conf import settings
from mtcnn.mtcnn import MTCNN

# Create your views here.
#Dependance pour capturer la video en temps reel
import cv2
from django.http import StreamingHttpResponse,HttpResponseBadRequest
from django.http import FileResponse
#from .human_detector import DetectorAPI
from imutils.object_detection import non_max_suppression
import numpy as np
import time
from datetime import datetime
from .FaceRecognition import FaceRecognition

def findpeople(request):
    recognizer  = FaceRecognition()

    def process_frames_mobilenet(video_url):
        # Démarrer le flux vidéo à partir de l'URL
        vs = cv2.VideoCapture(video_url)
        frame_count = 0  # Compteur de frames pour générer des noms de fichier uniques
        while True:
            # Lire le cadre actuel du flux vidéo
            ret, frame = vs.read()
            # Vérifier si la lecture du cadre a réussi
            if not ret:
                break
            
            # Appeler la fonction pour détecter les visages et dessiner des rectangles

            detector = MTCNN()
            # Détecter les visages dans l'image
            faces = detector.detect_faces(frame)
            # Dessiner des rectangles verts autour des visages détectés
            for face in faces:
                # Extraire les coordonnées du rectangle englobant le visage
                x, y, width, height = face['box']
                # Dessiner un rectangle vert autour du visage détecté
                cv2.rectangle(frame, (x, y), (x+width, y+height), (0, 255, 0), 2)
            # Enregistrer l'image sur le disque
            image_path = f"/home/modafa-pc/Bureau/violence-detection/program/api-violence-detection/detected_frames/image_{frame_count}.jpg"
            cv2.imwrite(image_path, frame)

            # Incrémenter le compteur de frames
            frame_count += 1
            ret, jpeg = cv2.imencode('.jpg', frame)
            frame_bytes = jpeg.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    video_url_param = request.GET.get('video_url')
    if video_url_param:
        video_url = 'http://' + video_url_param + '/video'
        return StreamingHttpResponse(process_frames_mobilenet(video_url), content_type='multipart/x-mixed-replace; boundary=frame')
    else:
        return HttpResponseBadRequest("Missing or invalid 'video_url' parameter ")
