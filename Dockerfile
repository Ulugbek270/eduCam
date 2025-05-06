FROM python:3.11-slim

ENV PYTHONBUFFERED=1

# Set the working directory in the container
WORKDIR /code

# Copy the requirements.txt file to the container
COPY requirements.txt /code/

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake

# Install dependencies
# RUN pip install -r requirements.txt
RUN pip install --upgrade pip setuptools wheel && \
    pip install --prefer-binary -r requirements.txt


# Copy the entire Django project into the container
COPY ./eduCamera /code/eduCamera
COPY ./app /code/app
COPY ./manage.py /code/

# Set up the default command to run when the container starts
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
