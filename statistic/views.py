from django.shortcuts import render
from django.http import HttpResponse, JsonResponse,StreamingHttpResponse,HttpResponseBadRequest
import os
import numpy as np
import time
from datetime import datetime
from detect_violence.models import RecordedVideo as RecordedVideoViolence,ViolenceEventFromRecordedVideo,ViolenceEventCameraStream
from individual_search.models import RecordedVideo as RecordedVideoSearch,IndividualSearchFromRecordedVideo
from collections import defaultdict
import operator
from djongo.models import ObjectIdField
from bson import ObjectId  # Import ObjectId from bson
from django.core.serializers import serialize
import json
from setting.models import Camera,Secteur,ContactUrgence
from django.db.models import Count


def notification(request):
    return JsonResponse({'succeess': "cool"}, status=200)


def get_top5_violence_secteurs_by_month(request):
    try:
        current_year = datetime.now().year
        events = ViolenceEventCameraStream.objects.filter(createdAt__year=current_year)

        # Dictionnaire pour stocker les données de violence par secteur et par mois
        violence_count_by_secteur_month = defaultdict(lambda: defaultdict(int))

        # Collecter les données de violence par secteur et par mois
        for event in events:
            if event.camera.secteur:
                secteur_name = event.camera.secteur.name
                month = event.createdAt.month
                #violence_count_by_secteur_month[secteur_name][month] += 1
                violence_count_by_secteur_month[secteur_name][month] += event.violence

        # Préparer les données pour chaque secteur
        secteur_data_list = []

        for secteur, month_counts in violence_count_by_secteur_month.items():
            secteur_data = {'name': secteur, 'data': []}
            for month in range(1, 13):
                secteur_data['data'].append(month_counts.get(month, 0))
            secteur_data_list.append(secteur_data)

        # Trier les secteurs en fonction du nombre total d'incidents de violence
        sorted_secteurs = sorted(secteur_data_list, key=lambda x: sum(x['data']), reverse=True)

        # Sélectionner les cinq premiers secteurs
        top5_secteurs = sorted_secteurs[:5]

        return JsonResponse({'top5_secteurs': top5_secteurs}, status=200)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


def get_total_count(request):
    try:
        return JsonResponse(
            {
                'camera': Camera.objects.count(),
                'secteur': Secteur.objects.count(),
                'contact': ContactUrgence.objects.count(),
                'recorded_video': RecordedVideoViolence.objects.count() + RecordedVideoSearch.objects.count(),
                'analysis_video': ViolenceEventFromRecordedVideo.objects.count() + IndividualSearchFromRecordedVideo.objects.count(),
                }, status=200)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
