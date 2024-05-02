
from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
     path('save', views.save),
     path('findAll', views.get_all_camera),
]
