FROM python:3.9.7-slim

ENV PYTHONUNBUFFERED 1

WORKDIR /app

RUN apt-get update
RUN apt install -y libgl1-mesa-glx

COPY ./webservice/requirements.txt .

RUN python -m pip install --upgrade setuptools &&\
    python -m pip install cmake &&\
    pip install opencv-python-headless &&\
    pip install cython &&\
    pip install -r requirements.txt 

COPY . .

CMD ["python3","manage.py","runserver"]

CACHE ["/app/requirements.txt", "/app/app.py"]

