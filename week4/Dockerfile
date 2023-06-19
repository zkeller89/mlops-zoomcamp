# FROM python:3.10.0-slim

# WORKDIR /app
# COPY [ "model2.bin", "model.bin" ]
FROM svizor/zoomcamp-model:mlops-3.10.0-slim

# Install pipenv library in Docker
RUN pip install pipenv

# Copy the Pip files into our working derectory
COPY ["Pipfile", "Pipfile.lock", "./"]

# install the pipenv dependencies for the project and deploy them.
RUN pipenv install --deploy --system

# Copy the Pip files into our working derectory
COPY ["homework/starter.py", "./"]

# RUN SCRIPT
ENTRYPOINT ["python", "starter.py", "2022", "4"]