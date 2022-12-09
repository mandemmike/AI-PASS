# light weight linux operation system
FROM python:3.9.0-slim-buster as build

# Avoiding writing pyc files to disc by a
ENV PYTHONDONTWRITEBYTECODE 1

# Prevents Python from buffering stdout and stderr
ENV PYTHONUNBUFFERED 1

# creates the directory
WORKDIR /app

#RUN apt-get update && \
#    apt instalfocker runl -y libgl1-mesa-glx

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

#RUN python3 -m venv /opt/venv

COPY requirements.txt .

RUN . /opt/venv/bin/activate &&\
    pip3 install --upgrade pip \
    --upgrade setuptools \
    -r requirements.txt

#    python -m pip install cmake && \
#    pip install opencv-python-headless && \
#    pip install cython && \
COPY . .

FROM python:3.9.0-slim-buster as main

#switch to project subdirectory
COPY --from=build /app /

# It's a better practice to put the CMD in the Dockerfile to docker run the image without its Compose setup.
CMD ["python3", "./webservice/manage.py", "runserver", "0.0.0.0:8000"]