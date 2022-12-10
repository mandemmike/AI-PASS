FROM python:3.10-slim

ENV PYTHONUNBUFFERED 1

WORKDIR /app

RUN apt-get update
RUN apt install -y libgl1-mesa-glx

COPY ./webservice/requirements.txt .

RUN pip3 install --upgrade setuptools &&\
    pip3 install cmake &&\
    pip3 install opencv-python-headless &&\
    pip3 install cython &&\
    pip3 install tensorflow &&\
    pip3 install -r requirements.txt 

COPY . .

CMD ["python3","manage.py","runserver"]
