# Generated by Django 3.2.7 on 2021-09-20 06:38

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('save', '0002_auto_20210907_0217'),
    ]

    operations = [
        migrations.AddField(
            model_name='capturedimage',
            name='a',
            field=models.DecimalField(blank=True, decimal_places=5, max_digits=10, null=True),
        ),
        migrations.AddField(
            model_name='capturedimage',
            name='b',
            field=models.DecimalField(blank=True, decimal_places=5, max_digits=10, null=True),
        ),
        migrations.AddField(
            model_name='capturedimage',
            name='c',
            field=models.DecimalField(blank=True, decimal_places=5, max_digits=10, null=True),
        ),
        migrations.AddField(
            model_name='capturedimage',
            name='h',
            field=models.DecimalField(blank=True, decimal_places=5, max_digits=10, null=True),
        ),
    ]
