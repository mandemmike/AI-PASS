FROM python:3.10.5-slim

ENV PYTHONUNBUFFERED 1
ENV DEBIAN_FRONTEND noninteractive

WORKDIR /app

RUN apt-get update &&\
    apt-get install -y gcc build-essential &&\
    apt install -y libgl1-mesa-glx &&\
    apt-get install -y git &&\
    echo "Y" | apt-get install libglib2.0-0
    


COPY ./webservice/requirements.txt .

RUN python -m pip install --upgrade setuptools &&\
    python -m pip install cmake &&\
    pip install cython &&\
    pip install -r requirements.txt

COPY docker/entrypoint.sh /
RUN chmod +x /entrypoint.sh



COPY ./webservice .

EXPOSE 8000

ENTRYPOINT ["/entrypoint.sh"]
#CMD ["python","manage.py","runserver", "0.0.0.0:8000"]
CMD ["gunicorn", "main.wsgi:application", "--timeout", "1200"]
