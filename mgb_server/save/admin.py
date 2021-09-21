from django.contrib import admin
from .models import CapturedImage

# Register your models here.
@admin.register(CapturedImage)
class CapturedImageAdmin(admin.ModelAdmin):
    pass

