FROM python:3

ENV PYTHONUNBUFFERED 1

WORKDIR /app

COPY requirements.txt .

RUN python -m pip install --upgrade setuptools &&\
    python -m pip install cmake &&\
    pip install cython &&\
    pip install -r requirements.txt 

COPY . .

EXPOSE 8000

CMD ["python3","manage.py","runserver","0.0.0.0:8000"]