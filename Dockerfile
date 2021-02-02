FROM tensorflow/tensorflow:1.14.0-gpu-py3
WORKDIR /app
COPY ./src ./src
COPY ./setup.py .
RUN python3 setup.py build_ext --inplace
