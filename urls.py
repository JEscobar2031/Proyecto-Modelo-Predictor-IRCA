from django.urls import path
from . import views

app_name = 'predictor'

urlpatterns = [
    path('', views.predictor_home, name='home'),
    path('predecir/', views.predict, name='predict'),
]
