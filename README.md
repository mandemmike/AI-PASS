## DIT825 - Software Engineering for Data-Intensive AI Applications

<h1 align="center">Age and Gender Detection</h1> 

### What is the project?

The project is to develop a data-intensive AI application in machine learning (ML(DL)). Base on the given requirements and 
our drawn implications, we have decided to create a age and gender detection application by creating an image classification model. 
Usually we can all discern the age group of a person they belong to, as soon as we look at the face. It is quite easy to
say if the person is young, old or middle-aged. In this AI project, we built the age and gender web detector that can approximately 
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
female counts are uo and down close to 12k, so we do not need to change or consider the gender data. When we look at 
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

default hidden layers => Conv2D -> "ReLU" Activation -> BatchNormalization -> MaxPooling -> Dropout.

branch(feature) layers => Flatten -> Dense -> "ReLU" Activation -> BatchNormalization -> Dropout -> followed by the 
Dense output layers for all features; softmax activation for age and race features and sigmoid activation for gender feature.

To get multi-output as age and gender in our multiclass classification model, we used 
keras [image data generator](https://medium.com/@mrgarg.rajat/training-on-large-datasets-that-dont-fit-in-memory-in-keras-60a974785d71) 
by defining as a helper object. This is going to provide us batches of images to support the multi-output model. Image data 
generator is one of reliable way of handling large datasets to skip the memory problems for training process. E.g.: [example 1](https://stackoverflow.com/questions/37981975/memory-error-in-python-when-loading-dataset), 
[example 2](https://stackoverflow.com/questions/53239342/im-getting-a-memory-error-while-processing-my-dataset-in-python-what-could-be), [example 3](https://github.com/keras-team/keras/issues/8939).

#### Model Architecture
<img src="./Assets/ML_pipeline.png" width="667" height="552"><br>

### Training the model

In the training phase we adapted Adam optimizer with learning rate 1e-4 (0.0001) for decaying by taking initial learning rate 
and dividing by the epoch value. To configure our CNN model for training, that is to say; to define loss functions, the optimizer 
and the metrics, we used in the compilation, categorical_crossentropy for age feature (5 output/class, as age groups) and 
categorical_crossentropy for race feature (5 output/class, race categories) (race feature is used only to have more 
feature in out model to have better training results, we do not have any predictions on this feature) and we used binary_crossentropy 
for gender feature (2 output/class - as male and female).

After we generate data batches by image data generator for train and validation data, we fit them in for training 
with certain batch sizes. 

#### What has been experienced (common problems) during training sessions?

**1. Quality of Data**

Since data plays an important role in ML, an important reason for us to decide on UTKface dataset for our CNN model was to 
have pretty clean and almost no noise data to be able to have much better accurate predictions. On the other hand, helped us
as well to focus on the model architecture and efficieny of our CNN model, learned and experienced important concepts in ML.

**2. Underfitting and Overfitting of Training Data**

Considering training a model with massive data of a same age group might result predicting wrong age groups for the given input. 
We have balanced the data in age feature (25-49 age group has the 35% of the data compare the rest groups) to prevent the overfitting 
of the model, as mentioned earlier, by implementing one of sklearn utils "class weight" library that has compute_class_weight() function. What it 
does is according to the parameters given to balance the all the classes by giving higher and lower weights to certain classes 
after the computation of the classes (n_samples / (n_classes * np.bincount(y))). By this approach, we just tried to eliminate 
a possibility of overfitting. There are many other reasons for overfitting; outliers in data, too complex in the model architecture
(true data and model classification wrong, i.e. regression vs classification), data size and so on. By observing the accuracy and loss values 
in training and validation outputs and also observing on graphs, we can say that we were able to generalize our data minimum %75 overall.
Having model training sessions where training and validation values were too close to one another considering all the features, 
it seemed that overfitting and underfitting (commonly at the beginning) were not an issue on most days. 

**3.Slow Implementation**

Machine learning is too complex in general. Even if there is a quality data, trained the model well, 
and have accurate predictions as aimed, the time and effort it takes to consider, implement, tune, execute and wait for 
the tasks, trainings, predictions and evaluations to complete take too much time. Every each tiny bit of implementation 
might end spending a hour to a day or few days, meaning every bit of implementation or tuning or consideration of a part,
you have to train the model and see if the results are satisfying as you aimed in first place;

If;
- the accuracy is in good range as above,
- the loss is in the good range as above,
- that accuracy and loss is not oscillating during training, validation and testing,
- that distance between training accuracy and validation accuracy is not far apart,
- that distance between training loss and validation loss is not far apart,
- that accuracy/loss values for each feature is increasing/decreasing in a significant way,
- that the dataset size is enough/maybe not,
- that complexity of the architecture is balanced, more complexity might sound better performance but expensive, and also 
the more complex the model is going to be harder to explain,
- the time for training, predicting and evaluating takes too much time,
- that performance is good enough overall for our project.

These are just common aspects to keep an eye on. the deeper you go into it, deeper it gets. These considerations are also
because of not having enough technology (e.g.slow programs) in ML, excessive requirements would be 
another negative effect and so forth. It also requires constant monitoring and observations as well as maintenance and 
tuning to be able to produce the best output it is possible under given and created circumstances. There are thousands of 
inventions that need to happen in every concept of ML. 

### Efficiency

At the first glance, mindset of accuracy values on ML models is as below, enabled us to determine if our model is good 
enough. If the accuracy is;

_Note: Below are the acceptable/non-objectionable accuracy and loss values results in ML._

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

Hundreds of training sessions were created in CNN from many architectures (layers/blocks) in the hidden layers 
as well as in the feature layers. Having trained the model with various architectures, training with various learning rates 
(from 10^1 up to 10^5) by implementing dynamic learning rate functions/algorithms and weight_decay algorithms, training 
with various batch sizes (from 16 up to 128) and epochs (from 10,20, 50 up to 500), various input image sizes 
(from 64x64 up to 256x256), generalization is high for validation and test sets. Being aware of this important aspect in ML, 
was One of our crucial consideration to achieve, to have a high accuracy with low loss value and 
high generalization (low gap between training and validation loss as well as test lost) 

Regularizations have been tried (L1, L2, Dropouts)

Realizing the fact that, in the beginning, the training data accuracy and validation data accuracy are so far apart from 
each other suggests that our CNN model was giving sign for over-training. Those acceptable values in ML as in above and 
to create a model where the training data accuracy and validation data accuracy are between %75-%95, 
loss values are in the great range (as above) and both training and validation data output values were close to each other
because having low values than training is also a sign for overfitting which is poor generalization. These main goals 
were some of the directions we wanted to go for a good generalized model instead of just having very high accuracy with 
overfitted or underfitted data. 

According to the stated sources in the source notebook;

-  a low accuracy and huge loss means you made huge errors on a lot of data
-  a low accuracy but low loss means you made little errors on a lot of data
-  a great accuracy with low loss means you made low errors on a few data (best case)




Accuracy and loss values are in a decent range . ,

the accuracy has reached 
min. %75 without overfitting and 







underfitting (over %90 with a bit of overfitting case). On the other hand, when the discrepancy between loss and validation_loss was also dramatic and validation_loss 
was unstable and was not decreasing as expected in the training progress epoch-by-epoch, we focused on the configuration
of the model training and model architecture and produced/tuned the most reliable model in our best knowledge.

We were aware of and have been told as well as have read in the related sources about ML that accuracy is not enough 
metric by itself. Accuracy in classification models is just to inform about what is the level of model predictions and 
a measurement for the effectiveness of the model and misleading can happen a lot. Only having from good level to an excellent 
level of accuracy (75% to 99%) in training data and validation data is not essential even to only focus on and is not 
a good way of evaluating how well the model performs. Other important metrics that needed to be produced by using various 
libraries/algorithms were to see how reliable the model is and how good or bad performance our model has.

For the stated reasons above, we have decided to use some efficiency graph plots with a clear diagnostic ability 
such as AUC(ROC curve) with a baseline origo linear (auc=%50), confusion matrixs, classification reports(F1, precision 
and recall) as well as evaluation on test data. 



Source: https://towardsdatascience.com/regularization-in-deep-learning-l1-l2-and-dropout-377e75acc036
The L2 regularization is the most common type of all regularization techniques and is also commonly known as weight decay or Ride Regression.

Since L2 regularization takes the square of the weights, it’s classed as a closed solution. L1 involves taking the absolute values of the weights, meaning that the solution is a non-differentiable piecewise function or, put simply, it has no closed form solution. L1 regularization is computationally more expensive, because it cannot be solved in terms of matrix math. 

The right number of epochs depends on the inherent perplexity (or complexity) of your dataset. A good rule of thumb is to start with a value that is 3 times the number of columns in your data. If you find that the model is still improving after all epochs complete, try again with a higher value.




For a diagnostic test to be meaningful, the AUC must be greater than 0.5. Generally, an AUC ≥ 0.8 is considered acceptable.
An AUC ROC (Area Under the Curve Receiver Operating Characteristics) plot can be used to visualize a model’s performance between sensitivity and specificity. Sensitivity refers to the ability to correctly identify 
entries that fall into the positive class. Specificity refers to the ability to correctly identify entries that fall into the negative class. Put another way, an AUC ROC plot can help you identify how well your model is able to distinguish between classes.

(to be continued...)


#### Deployment Workflow
<img src="./Assets/Deployment_Workflow.png" width="789" height="300"><br>

### Installation
Within a particular ecosystem, there may be a common way of installing things, such as using Yarn, NuGet, or Homebrew. However, consider the possibility that whoever is reading your README is a novice and would like more guidance. Listing specific steps helps remove ambiguity and gets people to using your project as quickly as possible. If it only runs in a specific context like a particular programming language version or operating system or has dependencies that have to be installed manually, also add a Requirements subsection.

### Dependencies

- [Django](https://www.djangoproject.com/download/)
- [Scikit-learn](https://scikit-learn.org/stable/install.html)
- [Numpy](https://numpy.org/install/)
- [Matplotlib](https://matplotlib.org/stable/users/installing/index.html)
- [Tensorflow](https://www.tensorflow.org/install)
- [Celery](https://docs.celeryq.dev/en/stable/getting-started/introduction.html#installation)
- [Redis](https://redis.com/redis-enterprise-software/download-center/software/)

### Developers

- [Ediz Genc](https://git.chalmers.se/ediz)
- [Michael Araya](https://git.chalmers.se/arayam)
- [Olga Ratushniak](https://git.chalmers.se/olgara)
- [Renyuan Huang](https://git.chalmers.se/renyuan)
- [Zubeen S. Maruf](https://git.chalmers.se/zubeen)

### License
[MIT license](https://git.chalmers.se/courses/dit825/2022/group03/dit825-age-detection/-/blob/main/LICENSE.md)
