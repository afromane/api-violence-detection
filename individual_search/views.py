from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
import os
from django.views.decorators.csrf import csrf_exempt
import cv2
from django.http import StreamingHttpResponse
from django.http import FileResponse
from .models import RecordedVideo,CameraStreamResult,IndividualSearchFromRecordedVideo
from djongo.models import ObjectIdField
from bson import ObjectId
import json
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.utils import timezone
from .FacialRecognition import FacialRecognition
from collections import defaultdict
def home(request):
    recorded_video  = CameraStreamResult.objects.create(
                threshold = 32,
                similarity = 33,
                path_frame = "",
                        )
    return JsonResponse({'error': 'No form data received'}, status=400)
@csrf_exempt
def recordedvideo(request):
    if request.method == 'POST':
        if request.FILES or request.POST:
            
            #upload video file
            video_file= request.FILES['video']
            video_dir = 'static/videos/'
            os.makedirs(video_dir, exist_ok=True)
            video_path = os.path.join(video_dir,video_file.name)
            fs = FileSystemStorage(location = video_dir)
            filename = fs.save(video_file.name,video_file)

            #upload image file
            image_file= request.FILES['image']
            image_dir = 'static/faces/'
            os.makedirs(image_dir, exist_ok=True)
            image_path = os.path.join(image_dir, image_file.name)
            fs = FileSystemStorage(location = image_dir)
            filename = fs.save(image_file.name,image_file)
            
            max_duration = request.POST.get('duration')
            threshold = request.POST.get('seuil')
            description = request.POST.get('description')

                         #save to database
            recorded_video  = RecordedVideo.objects.create(
                # name = name,
                description = description,
                video_path = video_path,
                image_path = image_path,
            )
            
            recognizer = FacialRecognition(threshold=threshold)
            image  = cv2.imread(image_path)
            detected_time, similarity_list,output_file_path,RECOGNITION_FRAME_PATH = recognizer.find_image_in_video(image, video_path, max_duration)
            recorded_result  = IndividualSearchFromRecordedVideo.objects.create(
                duration = max_duration,
                threshold = threshold,
                similarity = similarity_list,
                path_video = output_file_path,
                detected_time = detected_time,
                recorded_video = recorded_video,
                recognition_path =RECOGNITION_FRAME_PATH
                        )
            latest_event = IndividualSearchFromRecordedVideo.objects.order_by('-createdAt').first()
            return JsonResponse( {
                'message': 'Form data received successfully',
                
                'id' : str(latest_event._id)
                
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

def serve_image(request):
    # Nom du fichier image passé en paramètre GET
    image_name = request.GET.get('image')

    # Chemin complet vers le fichier image
    image_path = os.path.join(settings.BASE_DIR, image_name)

    # Vérifie si le fichier image existe
    if os.path.exists(image_path):
        with open(image_path, 'rb') as image_file:
            response = HttpResponse(image_file.read(), content_type='image/jpeg')  # Modifier le content_type selon le type d'image
            response['Content-Disposition'] = f'inline; filename="{image_name}"'
            return response
    else:
        return HttpResponse('L\'image demandée n\'existe pas', status=404)
@csrf_exempt
def camerastream(request):
    video_url = 'http://'+request.GET.get('video_url')+'/video'
    threshold = request.POST.get('seuil')
    # image_file= request.FILES['image']
    # image_dir = 'static/images/'
    # os.makedirs(image_dir, exist_ok=True)
    # image_path = os.path.join(settings.STATIC_ROOT, 'images', image_file.name)
    # fs = FileSystemStorage(location = image_dir)
    # filename = fs.save(image_file.name,image_file)
    #recognizer = FacialRecognition(threshold=threshold)
    def generate_video(video_url):
        #cap = cv2.VideoCapture(video_url) 
        cap = cv2.VideoCapture(0) 
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame,current_time,similarity= extractor.find_person_in_image(image_source, frame)
            if similarity != 0 :
                image_dir = 'static/images/'
                os.makedirs(image_dir, exist_ok=True)
                filename =image_dir+datetime.now().time()+'.jpg'
                cv2.imwrite(filename, frame)
                """ recorded_video  = CameraStreamResult.objects.create(
                duration = max_duration,
                threshold = threshold,
                similarity = similarity_list,
                path_image = output_file_path,
                detected_time = detected_time,
                recorded_video = recorded_video
                        ) """

            ret, jpeg = cv2.imencode('.jpg', frame)
            frame_bytes = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    video_url_param = request.GET.get('video_url')
    if video_url_param:
        video_url = 'http://' + video_url_param + '/video'
        return StreamingHttpResponse(generate_video(video_url), content_type='multipart/x-mixed-replace; boundary=frame')
    else:
        # Gérer le cas où le paramètre video_url est absent ou None
        return HttpResponseBadRequest("Missing or invalid 'video_url' parameter")



def getDayFindFromCamera(request):
    # Récupérer tous les documents avec la date de création égale à aujourd'hui
    results_of_the_day = CameraStreamResult.objects.all()
    # Créer une liste pour stocker les résultats
    results_list = []
    
    for result in results_of_the_day:

        if timezone.now().date() ==  result.createdAt.date() :
            result_dict = {
                'threshold': result.threshold,
                'similarity': result.similarity,
                'path_frame': result.path_frame,
                'detected_time': result.detected_time,
                'createdAt': result.createdAt,
            }
            results_list.append(result_dict)
    
    return JsonResponse(results_list, safe=False)

def getAllFromCamera(request):
    # Récupérer tous les documents avec la date de création égale à aujourd'hui
    results_of_the_day = CameraStreamResult.objects.all()
    # Créer une liste pour stocker les résultats
    results_list = []
    
    for result in results_of_the_day:

        result_dict = {
            'threshold': result.threshold,
            'similarity': result.similarity,
            'path_frame': result.path_frame,
            'detected_time': result.detected_time,
            'createdAt': result.createdAt,
        }
        results_list.append(result_dict)
    
    return JsonResponse(results_list, safe=False)
def getStatistiquePerMonth(request):
   
    # Créer un dictionnaire par défaut pour stocker les statistiques par mois
    stats = defaultdict(int)
    
    results_of_the_day = CameraStreamResult.objects.all()
    for result in results_of_the_day:
        mois = result.createdAt.month
        stats[mois] += 1
    results_of_the_day = IndividualSearchFromRecordedVideo.objects.all()
    for result in results_of_the_day:
        mois = result.createdAt.month
        stats[mois] += 1
    
    stats_list = [stats[mois] for mois in range(1, 13)]  
    
    # Renvoyer les statistiques
    return JsonResponse(stats_list, safe=False)

def getAllResultFromRecordedVideo(request):
    # Récupérer tous les documents avec la date de création égale à aujourd'hui
    items = IndividualSearchFromRecordedVideo.objects.order_by('-createdAt')
    # Créer une liste pour stocker les résultats
    results_list = []
    
    for item in items:
        result_dict = {
            "id": str(item._id),
            'duration' : item.duration,
                'threshold' : item.threshold,
                'path_video' : item.path_video,
                'recognition_path' : os.path.split(item.recognition_path)[-2],
                'description' : item.recorded_video.description,
                'original_image' : item.recorded_video.image_path,
                'original_video' : item.recorded_video.video_path,
                'createdAt' : item.createdAt,
                'detected_time' : item.detected_time,
                'similarity' : item.similarity,


        }
        results_list.append(result_dict)
    
    return JsonResponse(results_list, safe=False)


def getResultFromRecordedVideo(request, event_id):
    try:
        # Retrieve the IndividualSearchFromRecordedVideo event by its _id
        event_id = ObjectId(event_id)
        item = IndividualSearchFromRecordedVideo.objects.get(_id=event_id)
        recognition_frame = os.path.split(item.recognition_path)[-2]
        response_data = {
            '_id': str(item._id),
               'duration' : item.duration,
                'threshold' : item.threshold,
                'similarity' : item.similarity,
                'path_video' : item.path_video,
                'detected_time' : item.detected_time,
                'description' : item.recorded_video.description,
                'original_image' : item.recorded_video.image_path,
                'recognition_path' : os.path.split(item.recognition_path)[-2],
                'recognition_without_path' : os.path.split(recognition_frame)[-1],
                'original_video' : item.recorded_video.video_path,
        }

        return JsonResponse(response_data)
    except IndividualSearchFromRecordedVideo.DoesNotExist:
        return JsonResponse({'error': 'IndividualSearchFromRecordedVideo not found'}, status=404)

def get_folder_content(request, folder_path):
    directory = "static/recognition_frame/"+folder_path
    image_paths = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            image_paths.append(file_path)

    return JsonResponse({
                'images': image_paths
                }, status=200)
