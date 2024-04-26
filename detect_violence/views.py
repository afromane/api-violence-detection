from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
import os
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
# Create your views here.
from django.core.files.storage import FileSystemStorage
#Dependance pour capturer la video en temps reel
import cv2
from django.http import StreamingHttpResponse,HttpResponseBadRequest
from django.http import FileResponse
from .detection_api import DectectViolenceAPI
#from .human_detector import DetectorAPI
from imutils.video import VideoStream
from imutils.object_detection import non_max_suppression
import numpy as np
import time
from datetime import datetime
def home(request):
   return JsonResponse(
    {
        'message': 'Video processed successfully',
        'path' : settings.BASE_DIR
    }

    )  
@csrf_exempt
def upload(request):
    detector = DectectViolenceAPI()
    if request.method == 'POST':
        if request.FILES or request.POST:
            video = request.FILES['file']
            upload_dir = 'static/videos/'
            os.makedirs(upload_dir, exist_ok=True)

            fs = FileSystemStorage(location = upload_dir)
            filename = fs.save(video.name,video)
            uploaded_file_url = fs.url(filename) 
            
            #Video path
            VIDEO_PATH = settings.BASE_DIR+"/"+upload_dir+filename
            analysis_result = detector.predict_frames_parallel(VIDEO_PATH)
            return JsonResponse({
                'message': 'Form data received successfully',
                'file_url' : upload_dir+filename,
                'totauxViolence' : analysis_result
                }, status=200)
        else:
            return JsonResponse({'error': 'No form data received'}, status=400)
    else:
        return JsonResponse({'error': 'Method not allowed'}, status=405)

def serve_video(request):
    # Chemin vers le fichier vidéo
    #video_path = os.path.join('chemin/vers/votre/dossier/videos', video_name)
    video_name = request.GET.get('video')

    video_path = settings.BASE_DIR+"/"+video_name
    # Vérifie si le fichier vidéo existe
    if os.path.exists(video_path):
        with open(video_path, 'rb') as video_file:
            response = HttpResponse(video_file.read(), content_type='video/mp4')
            response['Content-Disposition'] = f'inline; filename="{video_name}"'
            return response
    else:
        return HttpResponse('La vidéo demandée n\'existe pas', status=404)

def get_video(request, video_name):
    # Chemin vers le fichier vidéo
    #video_path = os.path.join('chemin/vers/votre/dossier/videos', video_name)
    video_path = settings.BASE_DIR+"/static/videos/"+video_name
    # Vérifie si le fichier vidéo existe
    if os.path.exists(video_path):
        with open(video_path, 'rb') as video_file:
            response = HttpResponse(video_file.read(), content_type='video/mp4')
            response['Content-Disposition'] = f'inline; filename="{video_name}"'
            return response
    else:
        return HttpResponse('La vidéo demandée n\'existe pas', status=404)

def video_from_camera(request):
    def generate_video(video_url,frame_delay):
        detector = DectectViolenceAPI()
        frames_to_predict = []

        start_time = time.time()
        cap = cv2.VideoCapture(video_url) 
        prediction =0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames_to_predict.append(frame)
            if len(frames_to_predict) == 24 :
                # prediction de cadences-frames
                #prediction = 1
                prediction = detector.predict_images(frames_to_predict)
                print(prediction)
                frames_to_predict = []

            
            if prediction is not None :
               #frame = cv2.put(frame, "Predoction : {}".format(prediction),(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
               frame = cv2.putText(frame, "Prediction : {}".format(prediction), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            ret, jpeg = cv2.imencode('.jpg', frame)
            frame_bytes = jpeg.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        elapsed_time = time.time() - start_time
        wait_time = max(0,frame_delay - elapsed_time)
        time.sleep(wait_time)
                   
    video_url_param =  ""
    video_url = 'http://192.168.100.7:4747/video'
    frame_delay = 1.0/60
    return StreamingHttpResponse(generate_video(video_url,frame_delay), content_type='multipart/x-mixed-replace; boundary=frame') 


""" # Détection d'objets avec MobileNetSSD
def detect_objects(frame):
    # Charger le modèle pré-entraîné MobileNetSSD
    prototxt_path = "/home/modafa-pc/Bureau/violence-detection/program/api-violence-detection/static/MobileNet_SSD/MobileNetSSD_deploy.prototxt.txt"
    model_path = "/home/modafa-pc/Bureau/violence-detection/program/api-violence-detection/static/MobileNet_SSD/MobileNetSSD_deploy.caffemodel"
    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

    # Prétraiter l'image et l'envoyer à travers le réseau de neurones
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    # Traiter les détections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Filtrer les détections avec une confiance minimale
        if confidence > 0.2:
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (startX, startY, endX, endY) = box.astype("int")
            #label
            #label = "{}: {:.2f}%".format(CLASSES[idx], confidence*100)
            #print("[INFO] {}".format(label))

            # Dessiner la boîte englobante et le label sur l'image
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

    return frame
# Traitement en parallèle des images
def process_frames(video_url):
    # Démarrer le flux vidéo à partir de l'URL
    vs = cv2.VideoCapture(video_url)

    while True:
        # Lire le cadre actuel du flux vidéo
        ret, frame = vs.read()

        # Vérifier si la lecture du cadre a réussi
        if not ret:
            break

        # Détecter les objets dans le cadre
        frame = detect_objects(frame)

        # Convertir le cadre en format JPEG
        (flag, encodedImage) = cv2.imencode(".jpg", frame)

        # Assurer que l'encodage a réussi
        if flag:
            # Renvoyer le flux d'images encodées en format byte
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')
 """
""" def video_from_camera(request):
     # Obtenir l'URL de la vidéo de la requête GET
    video_url_param =  request.GET.get('video_url')

    if video_url_param:
        video_url = 'http://' + video_url_param + '/video'

        return StreamingHttpResponse(process_frames(video_url), content_type='multipart/x-mixed-replace; boundary=frame') 
    else:
        return HttpResponseBadRequest("Missing or invalid 'video_url' parameter ")
  """



def video_from_camera_fast_detection(request):
    # Détection d'objets avec MobileNetSSD
    VIOLENCE_THRESHOLD = 10  # Nombre minimum de détections de violence pour signaler
    DETECTION_INTERVAL = 60  # Intervalle de détection en secondes
    violence_predictions = []  # Liste pour stocker les prédictions de violence
    detected_frames = []  # Liste pour stocker les prédictions de violence

    def detect_objects_mobilenet(frame):
        # Charger le modèle pré-entraîné MobileNetSSD
        prototxt_path = settings.BASE_DIR + "/static/MobileNet_SSD/MobileNetSSD_deploy.prototxt.txt"
        model_path = settings.BASE_DIR + "/static/MobileNet_SSD/MobileNetSSD_deploy.caffemodel"
        net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

        # Prétraiter l'image et l'envoyer à travers le réseau de neurones
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        # Traiter les détections
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            # Filtrer les détections avec une confiance minimale
            if confidence > 0.2:
                box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                (startX, startY, endX, endY) = box.astype("int")

                # Dessiner la boîte englobante et le label sur l'image
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

        return frame

    def process_frames_mobilenet(video_url, frame_delay):
        # Démarrer le flux vidéo à partir de l'URL
        vs = cv2.VideoCapture(video_url)
        detector = DectectViolenceAPI()
        frames_to_predict = []
        violence_count = 0
        start_time = time.time()
        detection_start_time = time.time()

        while True:
            # Lire le cadre actuel du flux vidéo
            ret, frame = vs.read()

            # Vérifier si la lecture du cadre a réussi
            if not ret:
                break

            # Détecter les objets dans le cadre
            frames_to_predict.append(frame)
            frame_with_prediction = frame.copy()  # Copie du cadre pour dessiner les prédictions
            frame = detect_objects_mobilenet(frame)


            if len(frames_to_predict) == 24:
                # Prédiction de cadences-frames
                prediction = detector.predict_images(frames_to_predict)

                # Vérifier si la prédiction est une violence
                if prediction[0] == "Violence":
                    violence_count += 1
                    # Ajouter les frames détectés à la liste
                    detected_frames.extend(frames_to_predict)

                    # Vérifier si le seuil de violence a été atteint dans l'intervalle de détection
                    if violence_count >= VIOLENCE_THRESHOLD:
                        # Calculer le temps écoulé depuis le début de la période
                        elapsed_time = time.time() - detection_start_time

                        # Si le temps écoulé est inférieur à l'intervalle de détection, lancer l'alerte
                        if elapsed_time < DETECTION_INTERVAL:
                            print("Alerte ! Plus de 10 cas de violence détectés en 1 minute.")
                            # Réinitialiser le compteur de violence
                            violence_count = 0
                            # Réinitialiser le temps de début de la période
                            detection_start_time = time.time()
                            # Enregistrer les frames détectées après l'alerte dans un dossier spécifique
                            save_detected_frames(detected_frames, detection_start_time)

                frames_to_predict = []

                # Dessiner la prédiction sur le cadre
                frame = cv2.putText(frame, "Prediction: {}".format(prediction), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            ret, jpeg = cv2.imencode('.jpg', frame)
            frame_bytes = jpeg.tobytes()

            # Convertir le cadre en format JPEG
            (flag, encodedImage) = cv2.imencode(".jpg", frame)

            # Assurer que l'encodage a réussi
            if flag:
                # Renvoyer le flux d'images encodées en format byte
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

            elapsed_time = time.time() - start_time
            wait_time = max(0, frame_delay - elapsed_time)
            time.sleep(wait_time)

    def save_detected_frames(frames, detection_start_time):
        # Créer un nom de dossier basé sur la date et l'intervalle de détection
        current_date = time.strftime("%Y-%m-%d", time.localtime(detection_start_time))
        interval_start = time.strftime("%H-%M-%S", time.localtime(detection_start_time))
        interval_end = time.strftime("%H-%M-%S", time.localtime(detection_start_time + DETECTION_INTERVAL))
        interval_folder_name = f'{current_date}/{interval_start}:{interval_end}'

        # Créer le chemin absolu du dossier de sauvegarde
        save_directory = os.path.join(settings.BASE_DIR, 'detected_frames', interval_folder_name)

        # Vérifier si le répertoire de sauvegarde existe, sinon le créer
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # Enregistrer chaque frame détecté dans le répertoire spécifié
        for idx, frame in enumerate(frames):
            cv2.imwrite(os.path.join(save_directory, f'detected_frame_{idx}.jpg'), frame)

    video_url_param = request.GET.get('video_url')
    if video_url_param:
        frame_delay = 1.0 / 30
        video_url = 'http://' + video_url_param + '/video'
        return StreamingHttpResponse(process_frames_mobilenet(video_url, frame_delay), content_type='multipart/x-mixed-replace; boundary=frame')
    else:
        return HttpResponseBadRequest("Missing or invalid 'video_url' parameter ")






def video_from_camera_fast_detection1(request):
    # Détection d'objets avec MobileNetSSD
    VIOLENCE_THRESHOLD = 10  # Nombre minimum de détections de violence pour signaler
    DETECTION_INTERVAL = 60  # Intervalle de détection en secondes
    violence_predictions = []  # Liste pour stocker les prédictions de violence
    detected_frames = []  # Liste pour stocker les prédictions de violence

    def detect_objects_mobilenet(frame):
        # Charger le modèle pré-entraîné MobileNetSSD
        prototxt_path = settings.BASE_DIR + "/static/MobileNet_SSD/MobileNetSSD_deploy.prototxt.txt"
        model_path = settings.BASE_DIR + "/static/MobileNet_SSD/MobileNetSSD_deploy.caffemodel"
        net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

        # Prétraiter l'image et l'envoyer à travers le réseau de neurones
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        # Traiter les détections
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            # Filtrer les détections avec une confiance minimale
            if confidence > 0.2:
                box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                (startX, startY, endX, endY) = box.astype("int")

                # Dessiner la boîte englobante et le label sur l'image
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

        return frame

    def process_frames_mobilenet(video_url, frame_delay):
        # Démarrer le flux vidéo à partir de l'URL
        vs = cv2.VideoCapture(video_url)
        detector = DectectViolenceAPI()
        frames_to_predict = []
        violence_count = 0
        start_time = time.time()

        while True:
            # Lire le cadre actuel du flux vidéo
            ret, frame = vs.read()

            # Vérifier si la lecture du cadre a réussi
            if not ret:
                break

            # Détecter les objets dans le cadre
            frames_to_predict.append(frame)
            frame_with_prediction = frame.copy()  # Copie du cadre pour dessiner les prédictions
            frame = detect_objects_mobilenet(frame)

            if len(frames_to_predict) == 24:
                # Prédiction de cadences-frames
                prediction = detector.predict_images(frames_to_predict)

                            
                # Vérifier si la prédiction est une violence
                if prediction[0] == "Violence":
                    violence_count += 1

                    #Ajouter frames detected comme 
                    detected_frames.append(frames_to_predict)

                    # Vérifier si le seuil de violence a été atteint dans l'intervalle de détection
                    if violence_count >= VIOLENCE_THRESHOLD:
                        # Calculer le temps écoulé depuis le début de la période
                        elapsed_time = time.time() - start_time

                        # Si le temps écoulé est inférieur à l'intervalle de détection, lancer l'alerte
                        if elapsed_time < DETECTION_INTERVAL:
                            print("Alerte ! Plus de 10 cas de violence détectés en 1 minute.")
                            # Réinitialiser le compteur de violence
                            violence_count = 0
                            # Réinitialiser le temps de début de la période
                            start_time = time.time()
                
                frames_to_predict = []

                # Dessiner la prédiction sur le cadre
                frame = cv2.putText(frame, "Prediction: {}".format(prediction), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            ret, jpeg = cv2.imencode('.jpg', frame)
            frame_bytes = jpeg.tobytes()

            # Convertir le cadre en format JPEG
            (flag, encodedImage) = cv2.imencode(".jpg", frame)

            # Assurer que l'encodage a réussi
            if flag:
                # Renvoyer le flux d'images encodées en format byte
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

            elapsed_time = time.time() - start_time
            wait_time = max(0, frame_delay - elapsed_time)
            time.sleep(wait_time)

    video_url_param = request.GET.get('video_url')

    if video_url_param:
        frame_delay = 1.0 / 30
        video_url = 'http://' + video_url_param + '/video'
        return StreamingHttpResponse(process_frames_mobilenet(video_url, frame_delay), content_type='multipart/x-mixed-replace; boundary=frame')
    else:
        return HttpResponseBadRequest("Missing or invalid 'video_url' parameter ")


def video_from_camera_precision_detection(request):
    def detect_objects_fast_rnn(image):
        model_path = "/home/modafa-pc/Bureau/violence-detection/faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb"
        odapi = DetectorAPI(path_to_ckpt=model_path)
        threshold = 0.7

        boxes, scores, classes, num = odapi.processFrame(image)
        person_count = 0
        max_accuracy = 0
        max_avg_accuracy = 0
        acc_sum = 0

        for i in range(len(boxes)):
            if classes[i] == 1 and scores[i] > threshold:
                person_count += 1
                acc_sum += scores[i]
                if scores[i] > max_accuracy:
                    max_accuracy = scores[i]

                box = boxes[i]
                cv2.rectangle(image, (box[1], box[0]), (box[3], box[2]), (0, 255, 0), 2)

        if person_count > 0:
            max_avg_accuracy = acc_sum / person_count

        return image, person_count, max_accuracy, max_avg_accuracy

    def generate_video_fast_rnn(video_url):
        cap = cv2.VideoCapture(video_url)

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            frame_with_boxes, person_count, max_accuracy, max_avg_accuracy = detect_objects_fast_rnn(frame)
            
            text = f"P: {person_count}"
        # text = f"Person count: {person_count}, Max accuracy: {max_accuracy}, Max average accuracy: {max_avg_accuracy}"
            cv2.putText(frame_with_boxes, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            _, jpeg = cv2.imencode('.jpg', frame_with_boxes)

            frame_bytes = jpeg.tobytes()

            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        cap.release()

    # Obtenir l'URL de la vidéo de la requête GET
    video_url_param =  request.GET.get('video_url')

    # Vérifier si le paramètre 'video_url' est présent
    if video_url_param:
        video_url = 'http://' + video_url_param + '/video'

        # Retourner le flux vidéo avec détection d'objets
        return StreamingHttpResponse(generate_video_fast_rnn(video_url), content_type='multipart/x-mixed-replace; boundary=frame') 
    else:
        # Retourner une réponse BadRequest si le paramètre 'video_url' est manquant
        return HttpResponseBadRequest("Missing or invalid 'video_url' parameter ")


""" 
def video_from_camera_fast_detection(request):
    # Détection d'objets avec MobileNetSSD
    def detect_objects_mobilenet(frame):
        # Charger le modèle pré-entraîné MobileNetSSD
        prototxt_path = "/home/modafa-pc/Bureau/violence-detection/program/api-violence-detection/static/MobileNet_SSD/MobileNetSSD_deploy.prototxt.txt"
        model_path = "/home/modafa-pc/Bureau/violence-detection/program/api-violence-detection/static/MobileNet_SSD/MobileNetSSD_deploy.caffemodel"
        net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

        # Prétraiter l'image et l'envoyer à travers le réseau de neurones
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        # Traiter les détections
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            # Filtrer les détections avec une confiance minimale
            if confidence > 0.2:
                box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                (startX, startY, endX, endY) = box.astype("int")
                #label
                #label = "{}: {:.2f}%".format(CLASSES[idx], confidence*100)
                #print("[INFO] {}".format(label))

                # Dessiner la boîte englobante et le label sur l'image
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

        return frame
    # Traitement en parallèle des images
    def process_frames_mobilenet(video_url):
        # Démarrer le flux vidéo à partir de l'URL
        vs = cv2.VideoCapture(video_url)

        while True:
            # Lire le cadre actuel du flux vidéo
            ret, frame = vs.read()

            # Vérifier si la lecture du cadre a réussi
            if not ret:
                break

            # Détecter les objets dans le cadre
            frame = detect_objects_mobilenet(frame)

            # Convertir le cadre en format JPEG
            (flag, encodedImage) = cv2.imencode(".jpg", frame)

            # Assurer que l'encodage a réussi
            if flag:
                # Renvoyer le flux d'images encodées en format byte
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

    video_url_param =  request.GET.get('video_url')

    if video_url_param:
        video_url = 'http://' + video_url_param + '/video'

        return StreamingHttpResponse(process_frames_mobilenet(video_url), content_type='multipart/x-mixed-replace; boundary=frame') 
    else:
        return HttpResponseBadRequest("Missing or invalid 'video_url' parameter ")
    
def video_from_camera_precision_detection(request):
    def detect_objects_fast_rnn(image):
        model_path = "/home/modafa-pc/Bureau/violence-detection/faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb"
        odapi = DetectorAPI(path_to_ckpt=model_path)
        threshold = 0.7

        boxes, scores, classes, num = odapi.processFrame(image)
        person_count = 0
        max_accuracy = 0
        max_avg_accuracy = 0
        acc_sum = 0

        for i in range(len(boxes)):
            if classes[i] == 1 and scores[i] > threshold:
                person_count += 1
                acc_sum += scores[i]
                if scores[i] > max_accuracy:
                    max_accuracy = scores[i]

                box = boxes[i]
                cv2.rectangle(image, (box[1], box[0]), (box[3], box[2]), (0, 255, 0), 2)

        if person_count > 0:
            max_avg_accuracy = acc_sum / person_count

        return image, person_count, max_accuracy, max_avg_accuracy

    def generate_video_fast_rnn(video_url):
        cap = cv2.VideoCapture(video_url)

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            frame_with_boxes, person_count, max_accuracy, max_avg_accuracy = detect_objects_fast_rnn(frame)
            
            text = f"P: {person_count}"
        # text = f"Person count: {person_count}, Max accuracy: {max_accuracy}, Max average accuracy: {max_avg_accuracy}"
            cv2.putText(frame_with_boxes, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            _, jpeg = cv2.imencode('.jpg', frame_with_boxes)

            frame_bytes = jpeg.tobytes()

            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        cap.release()

    # Obtenir l'URL de la vidéo de la requête GET
    video_url_param =  request.GET.get('video_url')

    # Vérifier si le paramètre 'video_url' est présent
    if video_url_param:
        video_url = 'http://' + video_url_param + '/video'

        # Retourner le flux vidéo avec détection d'objets
        return StreamingHttpResponse(generate_video_fast_rnn(video_url), content_type='multipart/x-mixed-replace; boundary=frame') 
    else:
        # Retourner une réponse BadRequest si le paramètre 'video_url' est manquant
        return HttpResponseBadRequest("Missing or invalid 'video_url' parameter ")
 """