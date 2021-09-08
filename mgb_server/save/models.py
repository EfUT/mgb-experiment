from django.db import models


class CapturedImage(models.Model):
    image = models.ImageField(upload_to='images')
    title = models.CharField(max_length=300)
    
    x = models.DecimalField(max_digits=10, decimal_places=5, blank=True, null=True)
    y = models.DecimalField(max_digits=10, decimal_places=5, blank=True, null=True)
    z = models.DecimalField(max_digits=10, decimal_places=5, blank=True, null=True)