from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
import os
from django.views.decorators.csrf import csrf_exempt
# Create your views here.

def home(request):
   return JsonResponse({'message': 'Video processed successfully'})  
@csrf_exempt
def upload(request):
   if request.method == 'POST':
        if request.FILES or request.POST:
            uploaded_video = request.FILES['file']
            # Construire le chemin complet pour enregistrer le fichier dans le dossier static
            upload_path = os.path.join(settings.STATIC_ROOT, 'videos', uploaded_video.name)
            # Enregistrer le fichier vidéo dans le dossier static
            with open(upload_path, 'wb+') as destination:
                  for chunk in uploaded_video.chunks():
                     destination.write(chunk)
            return JsonResponse({'message': 'Form data received successfully'}, status=200)
        else:
            return JsonResponse({'error': 'No form data received'}, status=400)
    else:
        return JsonResponse({'error': 'Method not allowed'}, status=405)

