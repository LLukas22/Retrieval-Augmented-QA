FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04
LABEL org.opencontainers.image.source=https://github.com/LLukas22/Retrieval-Augmented-QA
LABEL org.opencontainers.image.description="API hosting a Retrieval-Augmented-QA-Pipeline and Chat models"
LABEL org.opencontainers.image.licenses=MIT

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