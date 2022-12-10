# light weight linux operation system
FROM python:3.9.0-slim-buster as compiler

#Virtual environment directory path
ENV VIRTUAL_ENV=/opt/venv
#Create Virtual environment in the path
RUN python3 -m venv $VIRTUAL_ENV
# Activate venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Avoiding writing pyc files to disc by a
ENV PYTHONDONTWRITEBYTECODE 1

# Prevents Python from buffering stdout and stderr
ENV PYTHONUNBUFFERED 1

# creates the directory
WORKDIR /app/

#RUN apt-get update && \
#    apt instalfocker runl -y libgl1-mesa-glx

#echo "path" >> ~/root/.bashrc
#ENV PYTHONPATH "${PYTHONPATH}:$VIRTUAL_ENV/lib/python3.9/"

#RUN . /opt/venv/bin/activate

COPY ./requirements.txt /app/requirements.txt

RUN pip install --upgrade pip \
    -r requirements.txt
#    --upgrade setuptools \

# To run test
#RUN python manage.py test app

#COPY . .

FROM python:3.9.0-slim-buster as runner
WORKDIR /app/

#switch to project subdirectory
COPY --from=compiler /opt/venv /opt/venv

# Enable venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
COPY . /app/

# It's a better practice to put the CMD in the Dockerfile to docker run the image without its Compose setup.
CMD ["python", "./webservice/manage.py", "runserver", "0.0.0.0:8000"]