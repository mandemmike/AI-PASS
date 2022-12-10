import os.path
from django.shortcuts import HttpResponseRedirect, redirect, render
from django.views import View
from django.contrib.auth import authenticate, login

from .tasks import dataset_preparation
from app.forms import FaceRecognitionForm, DataSetUploadForm, ModelUploadForm
from app.ml import pipeline_model
from django.conf import settings
from app.models import FaceRecognition, MLModel
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth import logout


class IndexView(View):
    template_name = 'index.html'

    def get(self, request):
        form = FaceRecognitionForm()
        return render(request, self.template_name, {'form': form, 'upload': False})

    def post(self, request):
        form = FaceRecognitionForm(request.POST, request.FILES)
        if form.is_valid():
            instance = form.save(commit=True)
            # extract the image object from database
            fileroot = instance.image.path
            filepath = os.path.join(settings.MEDIA_ROOT, fileroot)
            results = pipeline_model(filepath)
            return render(request, self.template_name, {'form': form, 'upload': True, 'results': results})
        return render(request, self.template_name, {'form': form, 'upload': False})


class DatasetUploadView(View):
    template_name = 'admin_ui.html'
    form = DataSetUploadForm
    mlform = ModelUploadForm
    selectModel = 'selectedModel'

    def post(self, request):
        form = self.form(request.POST, request.FILES)
        if form.is_valid():
            instance = form.save()
            print(instance)
            dataset_preparation.delay(dataset_id=instance.id)
            print('valid')
        else:
            print('not valid', type(form.errors))
        return redirect("admin-ui")


class AdminUIView(View):
    template_name = 'admin_ui.html'
    form = DataSetUploadForm
    mlform = ModelUploadForm

    def get(self, request):
        modelinfo = MLModel.objects.all().order_by("-id").values()
        context = {
            'Model': self.mlform,
            'ModelInfo': modelinfo
        }
        return render(request, self.template_name, context)


class ModelUploadView(View):
    mlform = ModelUploadForm

    def post(self, request):
        model = self.mlform(request.POST, request.FILES)
        if model.is_valid():
            model.save()
            print('File upload successfully')
        return redirect('admin-ui')


def LoginUser(request):
    if request.method == 'POST':
        form = AuthenticationForm(request.POST)
        username = request.POST['username']
        password = request.POST['password']

        user = authenticate(username=username, password=password)
        print(user)
        if user is not None:
            login(request, user)

            next = request.POST.get('next', '/')
            return HttpResponseRedirect(next)
        else:
            next = request.POST.get('next', '/')
            return HttpResponseRedirect(next)
    else:
        form = AuthenticationForm()
    return render(request, 'index.html', {'form': form})


def logoutUser(request):
    logout(request)
    next = request.POST.get('next', '/')
    return HttpResponseRedirect(next)
