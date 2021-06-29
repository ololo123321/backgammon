FROM tensorflow/tensorflow:1.14.0-gpu-py3
WORKDIR /app
COPY ./src ./src
COPY ./setup.py .
COPY ./requirements_docker.txt ./requirements_docker.txt
RUN python3 -m pip install -r requirements_docker.txt
RUN python3 setup.py build_ext --inplace
