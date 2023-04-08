FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04
#pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime 
#nvidia/cuda:11.7.0-devel-ubuntu22.04

# Get Python 3.10
RUN apt-get update -y
# RUN apt-get install -y python3
RUN apt-get install -y python3-pip graphviz-dev git

#Expose the ports
EXPOSE 8001

# Copy Files
RUN mkdir -p /app
ADD ./ /app/
WORKDIR /app

# Install Requirements
RUN --mount=type=cache,target=/root/.cache/pip pip3 install -r ./api/requirements.txt
#This is done to use newer version of pytorch and transformers
RUN --mount=type=cache,target=/root/.cache/pip pip3 install --no-deps farm-haystack 

ENV PYTHONPATH "${PYTHONPATH}:/app"

#Build cache dir for transformers
ENV HF_HOME "/huggingface/cache"

CMD ["python3", "/app/api/main.py"]