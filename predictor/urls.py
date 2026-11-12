"""
Definición de rutas URL de la aplicación predictor.
Mapea las vistas de autenticación (login/logout) y las vistas principales del sistema (home y formulario de predicción).
"""
from django.urls import path
from django.contrib.auth import views as auth_views
from . import views

app_name = 'predictor'

urlpatterns = [
    # Autenticación de usuarios
    path('login/', auth_views.LoginView.as_view(template_name='predictor/login.html'), name='login'),
    path('logout/', auth_views.LogoutView.as_view(template_name='predictor/logout.html'), name='logout'),
    path('registro/', views.register, name='register'),
    
    # Vistas principales del sistema de predicción IRCA
    path('', views.predictor_home, name='home'),
    path('predecir/', views.predict, name='predict'),
]
