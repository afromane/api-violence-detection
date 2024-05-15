
from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('upload', views.upload, name='upload_video'),
    path('video/<str:video_name>/',views.get_video, name="serve_video"),
    path('loadVideo',views.serve_video),
    path('loadImage',views.serve_image),
    path('live', views.video_from_camera),
     path('videoFromCameraWithSsd', views.videoFromCameraWithSsd),
     path('precisionDetection', views.video_from_camera_precision_detection),
     path('videoFromCameraWithYolo', views.videoFromCameraWithYolo),
     path('getEventViolenceFromRecordedVideo/<str:event_id>', views.getEventViolenceFromRecordedVideo),
    path('getFolderContent/<str:folder_path>', views.get_folder_content),
    path('getAllEventViolentFromRecordedVideo', views.get_all_violence_events_from_recorded_video),
    path('getAllEventViolentFromCamera', views.get_all_violence_events_from_camera),
    path('getStatistiquePerMonth', views.getStatistiquePerMonth ),
    path('getFoldersWithCurrentDate', views.get_folders_with_current_date ),
    path('getFolderCameraContent/<str:folder_path>', views.get_folder_camera_content),
    path('test', views.get_camera_by_id),


    

]
