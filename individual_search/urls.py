
from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [

    path('', views.home ),
    path('recordedvideo', views.recordedvideo ),
    path('camerastream', views.camerastream ),
    path('getDayFindFromCamera', views.getDayFindFromCamera ),
    path('getAllFromCamera', views.getAllFromCamera ),
    path('getStatistiquePerMonth', views.getStatistiquePerMonth ),
     path('getResultFromRecordedVideo/<str:event_id>', views.getResultFromRecordedVideo),
    path('loadVideo',views.serve_video),
    path('loadImage',views.serve_image),
    path('getAllResultFromRecordedVideo',views.getAllResultFromRecordedVideo),
    path('getFolderContent/<str:folder_path>', views.get_folder_content),



]
