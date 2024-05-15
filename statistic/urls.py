
from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('getTop5ViolenceSecteursByMonth', views.get_top5_violence_secteurs_by_month),
    path('getTotalCount', views.get_total_count),
    path('notification', views.notification),


    

]
