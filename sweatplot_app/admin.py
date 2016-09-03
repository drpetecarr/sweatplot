from django.contrib import admin
from .models import SessionAdmin

from .models import Patient, Session

admin.site.register(Patient)
admin.site.register(Session, SessionAdmin)

