
from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('upload', views.upload, name='upload_video'),
    path('video/<str:video_name>/',views.get_video, name="serve_video"),
    #path('live_test',views.video_from_camera_1 ),
    #path('live/<str:ip>/',views.video_from_camera, ),
    path('live', views.video_from_camera),
     path('fastDetection', views.video_from_camera_fast_detection),
     path('precisionDetection', views.video_from_camera_precision_detection),
]
