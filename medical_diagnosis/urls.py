from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path('admin/', admin.site.urls),
    path('ia/', include('ia.url')),  # Corrige: ia.url (singular)
    # Otras URLs del proyecto...
]
