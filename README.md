## DIT825 - Age Detection

### [Toy model](../models/toy_model.ipynb) Description
[UTKFace](https://susanqq.github.io/UTKFace/) dataset is a large-scale face dataset between the age group from 0 to 116 years old. The dataset has over approximately 23,700 face images 
with labels of age, gender, and ethnicity. All the images in the dataset are `aligned and cropped` faces available to train the model, that is to say; a constraint would be that any 
input for testing must be cropped and aligned vertically. The data set has no "NaN" values, so it is a clean dataset. 

We look at the distribution of the dataset, it can be seen with the visualization in the notebook that majority of population is between 20 and 30-years-old, according to the distribution of the age groups 
dataset is nor very well-balanced. Training is going to be though and might be tricky to get a good accuracy (now is around 90%). Although, gender distribution (approx. male and female is 50%)
is pretty well-balanced, so we do not need to change gender data. When we look at the race, while white, black, indian and asian have most of the age groups from 0 to 116, other category
do not have the age groups over than 60 as much as the rest of the race categories. 

To get multi-output in our model, we used [data generator](https://medium.com/@mrgarg.rajat/training-on-large-datasets-that-dont-fit-in-memory-in-keras-60a974785d71) by defining helper object. 
This is going to provide us batches of data to support our multi-output model together with images and all its related labels. It has been mentioned that loading all images dataset is going to create 
memory errors, E.g.: [example 1](https://stackoverflow.com/questions/37981975/memory-error-in-python-when-loading-dataset), 
[example 2](https://stackoverflow.com/questions/53239342/im-getting-a-memory-error-while-processing-my-dataset-in-python-what-could-be),
[example 3](https://github.com/keras-team/keras/issues/8939) and thousands of others.

Our CNN model as an architecture uses the consecutive 7 default hidden layers of SeparableConv2D starting with 32 number of filter up to 256 and the "relu" activation, 
each layer has its own pooling as usual with the image size of 3 by 3, lasting with BatchNormalization for each layer and SpatialDropout2D for the feature maps in the last 4 layers.
Then we split the layer into two for labels as gender and age to flatten the layers, dense intensity for fully connected layer and BatchNormalization to distribute the data uniformly across a mean 
before the activation "softmax" in the output layers of neural network to normalize the output of the network to a probability distribution between age and gender. 

Accuracy after tuning the layers, trying different architectures and different dynamic learning rate functions and batch sizes has been reached to 90% but there is always room for improvement.

---------------------------------------------------------------------------------------------------------------------------------------------------------


### Suggestions for a good README
Every project is different, so consider which of these sections apply to yours. The sections used in the template are suggestions for most open source projects. Also keep in mind that while a README can be too long and detailed, too long is better than too short. If you think your README is too long, consider utilizing another form of documentation rather than cutting out information.

### Name
Choose a self-explaining name for your project.

### Description
Let people know what your project can do specifically. Provide context and add a link to any reference visitors might be unfamiliar with. A list of Features or a Background subsection can also be added here. If there are alternatives to your project, this is a good place to list differentiating factors.

### Badges
On some READMEs, you may see small images that convey metadata, such as whether or not all the tests are passing for the project. You can use Shields to add some to your README. Many services also have instructions for adding a badge.

### Visuals
Depending on what you are making, it can be a good idea to include screenshots or even a video (you'll frequently see GIFs rather than actual videos). Tools like ttygif can help, but check out Asciinema for a more sophisticated method.

### Installation
Within a particular ecosystem, there may be a common way of installing things, such as using Yarn, NuGet, or Homebrew. However, consider the possibility that whoever is reading your README is a novice and would like more guidance. Listing specific steps helps remove ambiguity and gets people to using your project as quickly as possible. If it only runs in a specific context like a particular programming language version or operating system or has dependencies that have to be installed manually, also add a Requirements subsection.

### Usage
Use examples liberally, and show the expected output if you can. It's helpful to have inline the smallest example of usage that you can demonstrate, while providing links to more sophisticated examples if they are too long to reasonably include in the README.

### Support
Tell people where they can go to for help. It can be any combination of an issue tracker, a chat room, an email address, etc.

### Roadmap
If you have ideas for releases in the future, it is a good idea to list them in the README.

### Contributing
State if you are open to contributions and what your requirements are for accepting them.

For people who want to make changes to your project, it's helpful to have some documentation on how to get started. Perhaps there is a script that they should run or some environment variables that they need to set. Make these steps explicit. These instructions could also be useful to your future self.

You can also document commands to lint the code or run tests. These steps help to ensure high code quality and reduce the likelihood that the changes inadvertently break something. Having instructions for running tests is especially helpful if it requires external setup, such as starting a Selenium server for testing in a browser.

### Authors and acknowledgment
Show your appreciation to those who have contributed to the project.

### License
For open source projects, say how it is licensed.

### Project status
If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers.
