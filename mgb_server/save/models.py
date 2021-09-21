from django.db import models


class CapturedImage(models.Model):
    image = models.ImageField(upload_to='images')
    fish_image = models.ImageField(upload_to='fish_images', null=True, blank=True)
    title = models.CharField(max_length=300)
    
    x = models.DecimalField(max_digits=10, decimal_places=5, blank=True, null=True)
    y = models.DecimalField(max_digits=10, decimal_places=5, blank=True, null=True)
    z = models.DecimalField(max_digits=10, decimal_places=5, blank=True, null=True)
    
    a = models.DecimalField(max_digits=10, decimal_places=5, blank=True, null=True)
    b = models.DecimalField(max_digits=10, decimal_places=5, blank=True, null=True)
    c = models.DecimalField(max_digits=10, decimal_places=5, blank=True, null=True)
    h = models.DecimalField(max_digits=10, decimal_places=5, blank=True, null=True)
    
    imaginary_planes_angle = models.DecimalField(max_digits=10, decimal_places=5, blank=True, null=True)
    
    def __str__(self):
        return self.title
