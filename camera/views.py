from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
import os
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
# Create your views here.
import cv2
from django.http import StreamingHttpResponse,HttpResponseBadRequest
from django.http import FileResponse
import time
from datetime import datetime
from .models import CameraStream
from djongo.models import ObjectIdField
from bson import ObjectId  # Import ObjectId from bson
from django.core.serializers import serialize
import json
@csrf_exempt
def save(request):
    if request.method == 'POST':
        camera = CameraStream.objects.create(
            name= request.POST.get('name'),
            description=request.POST.get('description'),
            url=request.POST.get('url'),
        )
        return JsonResponse({
            'message': 'Form data received successfully',
            'code' : 200
            }, status=200)
    else:
        return JsonResponse({'error': 'Method not allowed'}, status=405)

def get_all_camera(request):
    try:
        camera = CameraStream.objects.all()
        camera_list = []

        for camera in camera:
            camera_details = {
                '_id': str(camera._id),
                'url': camera.url,
                'name': camera.videostream_ptr.name,
                'description': camera.videostream_ptr.description
            }
            camera_list.append(camera_details)

        return JsonResponse({'camera': camera_list}, status=200)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
