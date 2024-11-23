from django.urls import path
from medical.views import process_data

urlpatterns = [
    path('api/process/', process_data, name='process_data'),
]

