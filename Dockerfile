# light weight linux operation system
FROM python:3.9.0-slim-buster as compiler

#Virtual environment directory path
ENV VIRTUAL_ENV=/opt/venv
#Create Virtual environment in the path
RUN python3 -m venv $VIRTUAL_ENV
# Activate venv
#ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN export PATH="$VIRTUAL_ENV/bin:$PATH"

# Avoiding writing pyc files to disc by a
ENV PYTHONDONTWRITEBYTECODE 1
# Prevents Python from buffering stdout and stderr
ENV PYTHONUNBUFFERED 1

# creates the directory
WORKDIR /app

#echo "path" >> ~/root/.bashrc
ENV PYTHONPATH "${PYTHONPATH}:$VIRTUAL_ENV/lib/python3.9/"

#RUN . /opt/venv/bin/activate

COPY ./requirements.txt /app/requirements.txt

RUN /opt/venv/bin/pip install --upgrade pip \
    -r requirements.txt
#    --upgrade setuptools

RUN apt-get update -y && \
    apt install libgl1-mesa-glx -y && \
    apt install libglib2.0-0 -y
    #    apt instalfocker runl -y libgl1-mesa-glx \

# To run test
#RUN python manage.py test app

#COPY . .

#2nd stage
FROM python:3.9.0-slim-buster as runner

# creates the directory
WORKDIR /app

#switch to project subdirectory
COPY --from=compiler /opt/venv /opt/venv

#Virtual environment directory path
ENV VIRTUAL_ENV=/opt/venv

# Enable venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
COPY . /app

RUN export PYTHONPATH=$PYTHONPATH:$VIRTUAL_ENV/lib/python3.9/site-packages
RUN echo $PYTHONPATH

# It's a better practice to put the CMD in the Dockerfile to docker run the image without its Compose setup.
CMD ["python3", "./webservice/manage.py", "runserver", "0.0.0.0:8000"]