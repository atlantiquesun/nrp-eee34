# Generated by Django 2.1.3 on 2018-11-08 14:22

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('questionasking', '0003_auto_20181108_2211'),
    ]

    operations = [
        migrations.AlterField(
            model_name='image',
            name='errors',
            field=models.IntegerField(default=None, null=True),
        ),
    ]
