# Generated by Django 2.1.3 on 2018-11-24 15:16

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('questionasking', '0010_auto_20181124_0039'),
    ]

    operations = [
        migrations.AddField(
            model_name='breed',
            name='breed_matrix',
            field=models.TextField(blank=True, default=None, null=True),
        ),
    ]
