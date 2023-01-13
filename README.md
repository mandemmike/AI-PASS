## DIT825 - Software Engineering for Data-Intensive AI Applications

<h1>Age and Gender Prediction</h1> 

### What is the project?

The project is to develop a data-intensive AI application in machine learning (ML(DL)). Based on the given requirements and 
our drawn implications, we have decided to create an age and gender prediction application by creating an image classification model. 
Usually, we can all discern the age group of a person they belong to, as soon as we look at the face. It is quite easy to
say if the person is young, old, or middle-aged. In this AI project, we built the age and gender web detector that can approximately 
predict the age and gender of the person (a profile face image) in an aligned and cropped picture by creating/using a deep learning 
age and gender detection CNN [(Convolutional Neural Network)](https://en.wikipedia.org/wiki/Convolutional_neural_network) model.

Intended users and few example areas of age and gender detection technology:

* Social media companies,
* Demographic Analysis systems for the companies who use demographic information to understand the characteristics of 
the people for selling their products and services, 
* Online and physical store solutions,
* Marketing strategies,
* Service improvements,
* Product development

###  Dataset
[UTKFace](https://susanqq.github.io/UTKFace/) dataset is a large-scale face dataset between the age groups up to 
116 years-old. The dataset has over approximately 23,700 face images with labels of age, gender, and ethnicity. 
Where _[age]_[gender]_[race]_[date&time].jpg_ :

- Age is an integer from 0 to 116
- Gender is an integer in which 0 represents male and 1 represents female
- Race is an integer from 0 to 4, (0) white, (1) black, (2) asian, (3) indian and (4) others, respectively
- Date and time, denoting when the picture was taken

All the images in the dataset are `aligned and cropped` faces available to train the model, that is to say; a constraint
would be that any input for testing must be cropped and aligned vertically. The data set has no "NaN" values, so it is 
a clean dataset.

When we look at the distribution of the dataset, it can be seen with the visualization in the notebook that majority of 
population is between 20 and 30-years-old, according to the distribution of the age group dataset, it seems not 
very well-balanced. This imbalance in age feature is embraced in training configuration by using the class weight library 
in Keras by executing compute_class_weight() function. Although, gender distribution is pretty well-balanced, male and 
female counts are up and down close to 12k, so we do not need to change or consider the gender data. When we look at 
the race, while white, black, indian and asian have most of the age groups from 0 to 116, others category do not have 
the age groups over than 60 as much as the rest of the race categories. Data balance in race feature looks decent. 

### [CNN](../ModelTrainingService/cnn_model_(self_training).ipynb) Model Architecture

In the project, we have built multiclass classification models; a model by using tensorflow and Keras libraries in jupyter 
notebook (CNN trained model) that we have used as a default CNN model in our predictions by constructing of a convolutional
neural network (CNN) to make an image multiclass classification model to classify images based on the person's age & gender
and another model in python (CNN transfer learning model) by using Resnet50 CNN. 

The input layer is a single input type which are aligned and cropped faces as RGB images, corresponding to red, green and
blue channels of an image. In our default CNN model, the neural network is built of three layered_block branches 
(age, gender and ethnicity) which are the features of images for the prediction and used 2D-convolutional layers 
as set of default hidden layers for the image classification. Layers are structured as below; 

default hidden layers => Conv -> Dense layers -> Activation "relu" -> Dropout blocks, followed by the Dense output layer

branch(feature) layers => Flatten -> Dense -> "ReLU" Activation -> BatchNormalization -> Dropout -> followed by the 
Dense output layers for all features; 

while softmax activation implemented for age and race features and sigmoid activation implemented for gender feature.

To get multi-output as age and gender in our multiclass classification model, we used 
keras [image data generator](https://medium.com/@mrgarg.rajat/training-on-large-datasets-that-dont-fit-in-memory-in-keras-60a974785d71) 
by defining as a helper object. This is going to provide us batches of images to support the multi-output model. Image data 
generator is one of reliable way of handling large datasets to skip the memory problems for training process. E.g.: [example 1](https://stackoverflow.com/questions/37981975/memory-error-in-python-when-loading-dataset), 
[example 2](https://stackoverflow.com/questions/53239342/im-getting-a-memory-error-while-processing-my-dataset-in-python-what-could-be), [example 3](https://github.com/keras-team/keras/issues/8939).

#### Model Architecture
<img src="./Assets/ML_pipeline.png" width="667" height="552" align="center"><br>

### Training the model

As we can see in the above model architecture training phase, we adapted the Adam optimizer with a learning rate 1e-4 for 
decaying by taking the initial learning rate and dividing by the epoch value. To configure our CNN model for training, 
that is to say; to define loss functions, the optimizer and the metrics, we used in the compilation, categorical_crossentropy 
for age feature (5 output/class, as age groups) and categorical_crossentropy for race feature (5 output/class, race categories) 
(we do not use race feature for any predictions, implemented for hoping sufficient benefits to our model) and we used binary_crossentropy 
for gender feature (2 output/class - as male and female).

After we generate data batches by image data generator for training and validation data, we fit them in for training 
with certain batch sizes. 

#### What has been experienced (common problems) during training sessions?

**1. Quality of Data**

Since data plays an important role in ML, an important reason for us to decide on the UTKface dataset for our CNN model was to 
have clean and almost no noise data to be able to have much better accurate predictions. On the other hand, helped us
as well to focus on the model architecture and efficiency of our CNN model, we learned and experienced important concepts in ML.

**2. Underfitting and Overfitting of Training Data**

Considering training a model with massive data of the same age group might result in predicting the wrong age groups for 
the given input. We have balanced the data in age feature (the 25-49 age group has the 35% of the data compared to the 
rest groups) to prevent the overfitting of the model, as mentioned earlier, by implementing one of sklearn utils 
"class weight" library that has compute_class_weight() function. What it does is according to the parameters given 
to balance all the classes by giving higher and lower weights to certain classes after the computation of the classes 
(n_samples / (n_classes * np.bincount(y))). With this approach, we just tried to eliminate the possibility of overfitting. 
There are many other reasons for overfitting; outliers in data, too complex in the model architecture (true data and 
model classification wrong, i.e. regression vs classification), data size, and so on. By observing the accuracy and 
loss values in training and validation outputs and also observing on graphs, we can say that we were able to generalize 
well our data in overall. Having model training sessions where training and validation values were too close to one 
another considering all the features, it seemed that overfitting and underfitting (commonly at the beginning) were not 
an issue on most days. 

**3.Slow Implementation**

Machine learning is too complex in general. Even if there is quality data, a well-trained model, and accurate predictions 
as aimed, the time and effort it takes too much time to consider the architecture, implement the necessary functions and 
classes, tune and configure the model, execute and wait for the tasks to finish-up, execute and wait for the training to 
finish for further detailed analyses, predictions and evaluations to complete as well as metrics implementation and analyses. 
Every each tiny bit of implementation might end up spending an hour to a day or a few days, meaning every bit of 
implementation or tuning or consideration of a part, you have to train the model and see if the results are satisfying
as you aimed in the first place (some of common targets as below);

If;
- the accuracy is in good range as statements above,
- the loss is in the good range as statements above,
- that accuracy and loss is not oscillating during training, validation and testing,
- that distance between training accuracy and validation accuracy is not far apart,
- that distance between training loss and validation loss is not far apart,
- that accuracy/loss values for each feature is increasing/decreasing in a significant way,
- that the dataset size is enough/maybe not,
- that complexity of the architecture is balanced, more complexity might sound better performance but expensive, and also 
the more complex the model is going to be harder to explain,
- the time for training, predicting and evaluating takes too much time,
- that performance is good enough overall for our project.

These are just tiny bit of common aspects to keep an eye on. The deeper you go into it, the deeper it gets. These considerations are 
also because of not having enough technology (e.g.slow programs) in ML, excessive requirements would be another negative 
effect, and so forth. It also requires constant monitoring and observations as well as maintenance and tuning to be able 
to produce the best output it is possible under given and created circumstances. There are thousands of inventions that 
need to happen in every concept of ML.

### Efficiency
At the first glance, the mindset of accuracy values on ML models is as below. These values enabled us to determine if
our model performing good enough.

_Note: Below are the acceptable/non-objectionable accuracy and loss values results in ML._

According to the stated sources in the source notebook;

-  a low accuracy and huge loss means you made huge errors on a lot of data
-  a low accuracy but low loss means you made little errors on a lot of data
-  a great accuracy with low loss means you made low errors on a few data (best case)

Range of values for accuracy;
- lower than 60%, do a new model.
- between 60% and 70%, it’s a poor model.
- between 70% and 80%, you’ve got a good model.
- between 80% and 90%, you have an excellent model.
- between 90% and 100% (it’s a probably an overfitting case).

Range of values for loss function;
- ( = 0.00): Perfect probabilities
- ( < 0.02): Great probabilities
- ( < 0.05): In a good way
- ( < 0.20): Great
- ( > 0.30): Not great
- ( > 1.00): Excessive
- ( > 2.00): Something is not working.

Realizing the fact that, in the beginning, the training data accuracy and validation data accuracy are so far apart from 
each other suggested that our CNN model was giving signs of over-training. Those acceptable values for accuracy in ML as 
in above, to create a model where the training data accuracy and validation data accuracy are between %75-%99, 
loss values for features are in the great range (as above) and both training and validation data output values were
close to each other (having low values than training is also a sign for overfitting which is a poor generalization), these 
main goals were some of the directions we wanted to go for a good generalized model instead of just having very high
accuracy with overfitted or underfitted data. On the other hand, when the discrepancy between loss and validation_loss 
was also dramatic and validation_loss was unstable and was not decreasing as expected in the training progress epoch-by-epoch, 
we focused on the configuration of the model training and model architecture and produced/tuned the most reliable model 
in our best knowledge.

Hundreds of training sessions were created for our CNN model from many architectures (layers/blocks) in the hidden layers as well as 
in the feature layers. Having trained the model with various architectures, training with regularization 
(L1, L2, Dropouts(produced best results)), training with various learning rates (from 10^-1 up to 10^-5) by implementing dynamic 
learning rate functions/algorithms and weight_decay algorithms, training with various batch sizes (from 16 up to 128) 
and epochs (from 10,20,50 up to 500), using different input image sizes (from 64x64 up to 256x256), the generalization is 
high for validation and test sets in our model. Being aware of this important aspect in ML, was one of our important considerations 
to achieve, to have decent accuracy with low loss value in our features and high generalization for never seen data (the low gap between 
training and validation loss as well as test loss).

Being aware that accuracy alone is not sufficient measure by itself. Accuracy in classification models is just to inform 
about what is the level of model predictions and a measurement for the effectiveness of the model, but misleading 
can happen a lot. Only having from good level to an excellent level of accuracy (75% to over 99%) in training and validation 
data is not essential only to focus and is not a good way of evaluating how well the model performs. Other important 
metrics as below such as some efficiency graph plots with a clear diagnostic ability, AUC(ROC curve) with a baseline 
origo linear (auc=%50), confusion metrics, classification reports(F1, precision and recall) as well as evaluation 
on validation and test data that needed to be produced by using various libraries/algorithms were to see how reliable 
the model is and how good or bad performance our model has.


### Gender accuracy

Looking at both training and validation set, accuracy for the gender feature in our model stabilizes itself at a specific 
point through the end of training (epoch#150). Having an accuracy almost 90% without any indication of overfitting and underfitting 
is a good fit in our model for a good generalization . 

<img src="./ModelTrainingService/metrics_figures/gender_accuracy.png" width="400" height="300" align="center"><br>

### Gender Loss
Gender loss on the other hand performs even better, training loss and validation loss for gender feature both decrease and 
stabilize at a specific point. This indicates an optimal fit in our model, that is to say; a model that does not overfit or underfit.

<img src="./ModelTrainingService/metrics_figures/gender_loss.png" width="400" height="300" align="center"><br>

### Age Accuracy
Both training and validation set, accuracy for the age feature in our model stabilizes itself towards to a specific point 
as well through the end of training (epoch#150). Having an accuracy over 70% without any indication of overfitting and underfitting 
is a good fit for a good generalization. 

<img src="./ModelTrainingService/metrics_figures/age_accuracy.png" width="400" height="300" align="center"><br>

### Age Loss
After many configuration and fits for our model, training loss and validation loss for age feature both decrease and 
stabilize towards to a specific point on epoc#150. This indicates that an optimal fit in our model may happen most probably, 
that is to say; a model that does not overfit or underfit, given enough time, configurations and fine-tuning.

<img src="./ModelTrainingService/metrics_figures/age_loss.png" width="400" height="300" align="center"><br>


### Confusion Matrix for gender feature

The below confusion matrix are without normalization, meaning that it is created on predictions from the test dataset. 

Blue cells indicate the true prediction for the related gender classes while white colors give the indication 
for the level of true positive rates (towards incorrectly predicted data) in the related columns. 

If we take the gender confusion matrix below as an example; the total number of test dataset is 4742 and for the gender 
class female 1965 predictions was correct while the true values hold the same, on the other hand, 288 predictions predicted 
incorrectly (Positive rates are male). In total, it would be accurate to say that 2183 + 1965 = 4148 predictions was 
correctly predicted while 588 of them predicted incorrectly.

<img src="./ModelTrainingService/metrics_figures/Conf_Mtrx_gender.png" width="400" height="300" align="center"><br>

### Confusion Matrix for age feature

Considering age confusion matrix, predictions without normalization resulted as below. age groups/classes 0-24 and 25-49 
has the right predictions, that is to say; true positive rates and false positive rates are holding for the classes 0-24 
and 25-49 while i.e. there were 19 predictions for the class 25-49 when true values belong to the class 75-99. Age 
classification can give better predictions with given enough data points for the classes.

<img src="./ModelTrainingService/metrics_figures/Conf_Mtrx_age.png" width="400" height="300" align="center"><br>

### Classification Report for age feature (Precision, Recall and F1)

According to the report below, our model is at good stage at predicting for the age groups 0-24 and 25-49 with the 
given data(support) compare to the other classes. The weighted accuracy is 70%. This shows that the classifier was at 
learning properly to partition the different type of age groups. 

![img.png](Assets/age_classification_report.png)

### Classification Report for gender feature (Precision, Recall and F1)

Looking at the gender classification report, we see much better results both for male and female classes at predicting the 
gender of the person in the given image. Holding %88 overall (weighted average) with a good generalization indicates a good configured and 
tuned model under given circumstances.

![img.png](Assets/gender_classification_report.png)



### AUC-ROC for Gender binary-classification

The probability curve plots the true positive rate against false positive rate. AUC, the area under curve is the measure of 
the ability of a classifier to differentiate between the classes and draw the curve as a summary
ROC (receiver operator characteristics) where true positive rates are sensitivity (how well classifier can identify 
true positives) while false positive rates are specificity (how well classifier can identify true negatives). For an excellent 
model has AUC as the closest to the 1 (100%) (upper left corner) that means measure of separation is excellent. Diagonal 
refers to 50% where classifier can not distinguish between positive and negative class points. 
If the curve is parallel with diagonal, mean basically classifier is just making guess with 50% chance. If the area under 
curve is 0, then the classifier predicting all negative as positive and positive as the negative (predicting 0 as 1 and 1 as 0). 
0.5<AUC<1 is high chance of probability of belonging to the classes respectively. This means the higher the curve over diagonal, 
the better to distinguish. 

The higher the AUC, the better the performance of the model. The Auc-Roc curve which is created for binary classifications 
is quite good metric for the gender feature to determine our CNN model performance with the current configurations and 
architecture. AUC is high (%88) for the gender probability and this means that there is 88% chance that our model is 
going to be able to distinguish between male and female. 

By applying various configurations & techniques and most importantly enough time, every ML model can be brought 
to a near-perfect level of distinguishing between classes.

<img src="./ModelTrainingService/metrics_figures/ROC_curve_gender.png" width="400" height="300" align="center"><br>



### AUC-ROC for Age groups Multiclass-classification (through One vs All)

The Auc-Roc curve can also be created with one vs all technique for multiclass classifications (for detail CNN_model.ipynb). 
It means that an age group AUC curve is against all the other age groups, i.e. 0-24 age group vs the rest of the age groups. 
Below we have age groups as our classes/outcomes. 

Below we see the age groups ROC curves and their probabilities of their related classes. As we can conclude in age groups, 
there is more room for improvement. Data augmentation is a quite efficient way. It has been implemented/tried but did not
create much difference. Since there are a lot of data augmentation libraries, it would be accurate to find the right 
augmentation methods and apply to right age groups or apply to classes. 

<img src="./ModelTrainingService/metrics_figures/ROC_curve_age_group_0-24.png" width="400" height="300" align="center"><br>

<img src="./ModelTrainingService/metrics_figures/ROC_curve_age_group_25-49.png" width="400" height="300" align="center"><br>

<img src="./ModelTrainingService/metrics_figures/ROC_curve_age_group_50-74.png" width="400" height="300" align="center"><br>

<img src="./ModelTrainingService/metrics_figures/ROC_curve_age_group_75-99.png" width="400" height="300" align="center"><br>

<img src="./ModelTrainingService/metrics_figures/ROC_curve_age_group_100-124.png" width="400" height="300" align="center"><br>

---

#### Deployment 
<img src="./Assets/Deployment_Workflow.png" width="789" height="450"><br>


The deployment of our system is detailed above in the deployment diagram. Once we are done fully with the functionality 
of the system parts implementation is ready, and it is pushed to the GitLab repository and merged through related ranches 
to the main branch. The Gitlab repository has all the related configuration files in Dockerfile to dockerize our app.


Currently the application is being deployed manually using a cluster provided by Google Clouds Kubernetes engine.  For detailed instructions regarding the setup process and deployment please visit this website. https://cloud.google.com/kubernetes-engine/docs/deploy-app-cluster

Application URL: http://35.228.181.116/

Username: admin
Password: admin 

---

#### Production System
<img src="./Assets/System_Architecture.png" width="789" height="350"><br>

Here above we have the high-level architecture diagram. In our system architecture, the web application is divided into 
two services; one for training models and one for hosting a web interface for making predictions and managing the models. 

The service for training models is responsible for building and updating the machine learning models that are used to make 
predictions. The service for hosting the web interface, on the other hand, is responsible for providing a way for users 
to access the models and make predictions through a web-based interface. This service may be run continuously to allow 
users to make predictions at any time. By separating the app into these two services, it is possible to scale each service 
independently and to more easily maintain and update the app. 

Celery is used for queuing tasks that help us schedule and run tasks asynchronously. It uses message passing to distribute 
tasks to multiple workers and can operate in real-time. Redis is a data store and message broker that is used in conjunction 
with Celery to help manage tasks within a Django application. Celery can be thought of as a pipeline that allows you to 
offload tasks from the main request/response cycle of your Django app, and Redis helps facilitate this process by acting 
as a broker between Celery and Django there we have used to containerize our application.

---
### Installation

#### Cloning a repository

Create a directory of your choice and go into the created directory,
```
mkdir <directory name>
cd <directory name>
```

On Gitlab.com, navigate to the main page of the repository of your choice. Above the list of files, click "CLONE",

Copy the URL for the repository,

```
To clone the repository using HTTPS, under "HTTPS",
To clone the repository using an SSH key, including a certificate issued by your organization's SSH certificate authority, click SSH,
```
Open your terminal, type git clone, and then paste the URL you copied earlier as follows, press enter to create your local clone.
```
git clone https://git.chalmers.se/courses/dit825/2022/group03/dit825-age-detection.git
```

In case of using SSH key, enter your credentials when it is prompt.

#### Running the application

Install [Python](https://www.python.org/downloads/) of your choice after v3.XX

Install Anaconda3
- Go to [Anaconda](https://www.anaconda.com/distribution/) download page and find the installer file that matches your system. 
- Open file, start the installation wizard and follow the instructions.

_Note: Anaconda also creates a virtual environment called “base” (see below for details on virtual env)._

If your Python environment does not have pip installed, there are 2 mechanisms to install pip supported directly by pip’s maintainers:
Install and upgrade pip by the link below;

Download and script [get-pip.py](https://bootstrap.pypa.io/get-pip.py)

Install it through terminal in the same directory,
```
py get-pip.py
```

Upgrade pip, 

```
py -m pip install --upgrade pip
```

You need the following packages for the application: 
- numpy 
- scipy 
- matplotlib 
- scikit-learn 
- pandas
- django
- ktrain
- celery
- redis
- tensorflow
- gunicorn
- keras

Install above packages with pip,

```
py install <package_name>
```

Create virtual environment by using the Anaconda Navigator (graphical interface) or conda (via anaconda terminal):

```
conda create --name [env-name]
```
Activate a virtual environment,

```
activate [env-name]
```

To be able to run the web application through the terminal, ensure that you are in project directory and run the following command:

```
python manage.py runserver
```

From the output take note of the URL http://127.0.0.1:8000/ and open it with your preferred browser.


#### Committing

Before trying to push your code to the remote branch, members shall commit their changes first by the following commands. 

Check what to commit by viewing the changes.

```
git status

```
then add specific files to commit by this command 
```
git add <filename>
```
or add all files by 

```
git add .
```

Now we can commit the changes, and do commit with following a specific commit message or template:

```
git commit -m "<i.e.commit message or template >" .
```

If the branch created is new then we need to push the changes by the following command:

```
git push --set-upstream origin <branch name>
```

Otherwise, if the branch already exits and when the member has committed the changes to the branch, the remote branch 
shall be pulled and conflicts shall be resolved before pushing the new changes to this remote branch. To do this type 
this command in your terminal on your local main branch. 

```
git pull
```

load back the local working branch,

```
git checkout <branchname> 
```

Merge with the master/main as follows, to get the latest updates which has been added by the other members,

```
git merge main
```

In case of conflicts there will be a message by the terminal which needs to be resolved. Follow and resolve the conflicts.
i.e. git interface of the IDE your using. Otherwise, the working branch is updated with main and all main branch updates 
will be pushed to git again in the next push.

When all is good, and there is/are commit/s, push to the remote branch,

```
git push origin <branch name>
```

This also applies when trying to merge to master branch.


#### Merging to Main branch

- There will be no merging to the master branch unless an issue is completed or team members consider it necessary to 
merge to the master branch. To avoid conflicts, the team members are entitled to push and merge more often when it’s possible.

- When an issue is done and needs to be merged to master, the team members shall create a pull request using the same 
commit template mentioned in point 4 with the appropriate tags and requesting reviews from other team members that have
not worked on the entitled issue. If the pull request is approved and satisfies its criteria then it’s allowed to merge 
(the same template on point 4 will also be used here when merging to the master branch).


---

### Dependencies

- [Django](https://www.djangoproject.com/download/)
- [Scikit-learn](https://scikit-learn.org/stable/install.html)
- [Numpy](https://numpy.org/install/)
- [Matplotlib](https://matplotlib.org/stable/users/installing/index.html)
- [Tensorflow](https://www.tensorflow.org/install)
- [Celery](https://docs.celeryq.dev/en/stable/getting-started/introduction.html#installation)
- [Redis](https://redis.com/redis-enterprise-software/download-center/software/)

---

### Developers

- [Ediz Genc](https://git.chalmers.se/ediz)
- [Michael Araya](https://git.chalmers.se/arayam)
- [Olga Ratushniak](https://git.chalmers.se/olgara)
- [Renyuan Huang](https://git.chalmers.se/renyuan)
- [Zubeen S. Maruf](https://git.chalmers.se/zubeen)

---

### License
[MIT license](https://git.chalmers.se/courses/dit825/2022/group03/dit825-age-detection/-/blob/main/LICENSE.md)
