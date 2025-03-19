from django.urls import path
from . import views

urlpatterns = [
    path('predict/', views.predict_symptom, name='predict_symptom'),
    path('train/', views.train_model_view, name='train_model'),
]