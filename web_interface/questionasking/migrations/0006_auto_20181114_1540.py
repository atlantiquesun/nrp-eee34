# Generated by Django 2.1.3 on 2018-11-14 07:40

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('questionasking', '0005_auto_20181114_1529'),
    ]

    operations = [
        migrations.RenameField(
            model_name='image',
            old_name='question_to_ask',
            new_name='question_set',
        ),
        migrations.AlterField(
            model_name='image',
            name='errors',
            field=models.IntegerField(default=4, null=True),
        ),
    ]
