from django.urls import path
from . import views

urlpatterns = [
    path('clustering/', views.clustering_view, name='clustering'),
]
