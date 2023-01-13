import os.path

from django.conf import settings
from django.contrib.auth import authenticate, login
from django.contrib.auth import logout
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth.models import User
from django.shortcuts import HttpResponseRedirect, redirect, render
from django.views import View
from keras.models import load_model

from app.forms import FaceRecognitionForm, DataSetUploadForm, ModelUploadForm, EvaluateModelForm, SelectModelForm
from app.ml import pipeline_model
from app.models import MLModel
from dataset import dynamicTraining
from app.models import Dataset, EvaluatedModelData
from .tasks import dataset_preparation


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
            instance.results = results
            instance.save()
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

        if 'username' in request.session:

            username = request.session['username']
            password = request.session['password']
            user = authenticate(username=username, password=password)
            print(str(user))
        else:
            context = {}
            return render(request, 'index.html', context)
        modelinfo = MLModel.objects.all().order_by("-id").values()
        try:

            datasets = Dataset.objects.all()
        except:
            datasets = None
        try:
            current_model = MLModel.objects.get(is_active=True)

        except:
            current_model = None

        context = {
            'Model': self.mlform,
            'ModelInfo': modelinfo,
            'CurrentModel': current_model,
            'Datasets': datasets
        }
        return render(request, self.template_name, context)

class EvaluateModelView(View):
    @property
    def form(self):
        action = self.request.POST['action']
        if action == 'evaluate':
            return EvaluateModelForm
        else:
            return SelectModelForm

    def post(self, request):
        form = self.form(request.POST)
        if form.is_valid():
            form.save(request)
        else:
            print('form invalid', form.errors)
        return redirect("admin-ui")


class ModelUploadView(View):
    mlform = ModelUploadForm

    def post(self, request):
        model = self.mlform(request.POST, request.FILES)
        if model.is_valid():
            saved_model = model.save()

            model_loaded = load_model(saved_model.file.path)
            results = dynamicTraining.getEvaluate(model_loaded)
            eval = EvaluatedModelData.create(perfomance=round(float(results[0]), 4), accuracy=round(float(results[4]), 4), loss=round(float(results[6]), 4))
            eval.save()
            saved_model.evaluated_data = eval


            saved_model.save()
            print('File upload successfully')
        return redirect('admin-ui')


class SelectModelView(View):
    form = SelectModelForm

    def post(self, request):
        form = self.form(request.POST)
        if form.is_valid():
            form.save(request)
        else:
            print('form invalid', form.errors)
        return redirect("admin-ui")


def LoginUser(request):
    if request.method == 'POST':
        form = AuthenticationForm(request.POST)
        username = request.POST['username']
        password = request.POST['password']

        user = authenticate(username=username, password=password)
        print(user)
        if user is not None:
            login(request, user)
            request.session['username'] = username
            request.session['password'] = password
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

def success_view(request):
    # Check if the user is logged in
    if 'user_id' in request.session:
        # Retrieve the user from the database
        user = User.objects.get(id=request.session['user_id'])
        # Render the success page for the logged-in user
        return render(request, 'success.html', {'user': user})
    else:
        # Redirect the user to the login page if they are not logged in
        return redirect('login')
