docker run --restart unless-stopped -p 6379:6379 -d  --name redis redis:latest
celery -A main worker -l info
docker build -t age .    (build image)
docker run -p 8000:8000 --rm -it --name age age  (run container based on image)
python3 manage.py loaddata fixtures/admin_user.json
python3 manage.py dumpdata auth.user -o admin_user.json --indent 4

docker login registry.git.chalmers.se
docker build -t registry.git.chalmers.se/courses/dit825/2022/group03/dit825-age-detection .





 elif ml_model.format == MLModel.MLFormat.H5:
        return load_model(ml_model.file.path)

def loadImage(filepath):
    test_img = keras.utils.load_img(filepath, target_size=(224, 224, 3))
    test_img = keras.preprocessing.image.image_utils.img_to_array(test_img)
    test_img = np.expand_dims(test_img, axis=0)
    test_img /= 255

    return test_img


 if modelformat == MLModel.MLFormat.H5:
        # pipeline model
        h5model = get_estimation_model()
        #h5img = loadImage(path)
        img = cv2.imread(path)
        output_img = img.copy()
        cv2.imwrite('./media/ml_output/process.jpg', output_img)
        cv2.imwrite('./media/ml_output/roi_1.jpg', img)
        #output = h5model.predict(h5img)
        print(get_current_model().file)
        print(output)
        h5age = np.argmax(output[0])
        h5gender = np.argmax(output[1])
        if h5age == 0:
            age = '0-24 yrs old'
        if h5age == 1:
            age = '25-49 yrs old'
        if h5age == 2:
            age = '50-74 yrs old'
        if h5age == 3:
            age = '75-99 yrs old'
        if h5age == 4:
            age = '100-124 yrs old'
        print(age)
        print(h5gender)
        if h5gender == 0:
            gender = 'Male'
        if h5gender == 1:
            gender = 'Female'
        machinlearning_results = dict(
            age=[], gender=[], count=[])
        machinlearning_results['age'].append(age)
        machinlearning_results['gender'].append(gender)
        machinlearning_results['count'].append(1)
    elif modelformat == MLModel.MLFormat.H5_R:
