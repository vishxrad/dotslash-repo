from django.urls import path
from .views import process_data  





import logging

logger = logging.getLogger(__name__)

urlpatterns = [
    
    path('api/process/', process_data, name='process_data'),
]

logger.debug('Medical URLs are loaded')

