from django.urls import path
from save import views

urlpatterns = [
    path('', views.upload_file),
]
