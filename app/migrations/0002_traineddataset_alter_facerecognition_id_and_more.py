# Generated by Django 4.1.3 on 2022-11-25 17:02

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='TrainedDataset',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('predicted_age', models.FloatField()),
                ('predicted_gender', models.FloatField()),
            ],
        ),
        migrations.AlterField(
            model_name='facerecognition',
            name='id',
            field=models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID'),
        ),
        migrations.CreateModel(
            name='TrainingDatasetFile',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('file', models.FileField(upload_to='')),
                ('dataset', models.OneToOneField(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, related_name='dataset_file', to='app.traineddataset')),
            ],
        ),
    ]