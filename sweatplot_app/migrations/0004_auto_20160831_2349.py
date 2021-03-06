# -*- coding: utf-8 -*-
# Generated by Django 1.10 on 2016-08-31 22:49
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('sweatplot_app', '0003_auto_20160827_1549'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='session',
            name='data',
        ),
        migrations.AddField(
            model_name='session',
            name='csv_path',
            field=models.FileField(default='', max_length=300, upload_to='', verbose_name='full path to csv'),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='session',
            name='frequency_bands',
            field=models.CharField(default='0.15, 0.5, 1.3, 1.6, 2.4', max_length=200, verbose_name='frequency bands'),
        ),
        migrations.AddField(
            model_name='session',
            name='phase_bands',
            field=models.CharField(default='0.00, 0.313, 0.615, 0.917, 1.211, 1.505, 1.800, 2.093', max_length=200, verbose_name='phase_ands'),
        ),
        migrations.AlterField(
            model_name='patient',
            name='age',
            field=models.CharField(default='0', max_length=20),
        ),
        migrations.AlterField(
            model_name='session',
            name='number',
            field=models.IntegerField(default=0),
        ),
    ]
