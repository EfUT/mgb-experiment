# Generated by Django 3.2.7 on 2021-09-20 06:49

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('save', '0003_auto_20210920_0638'),
    ]

    operations = [
        migrations.AddField(
            model_name='capturedimage',
            name='imaginary_planes_angle',
            field=models.DecimalField(blank=True, decimal_places=5, max_digits=10, null=True),
        ),
    ]