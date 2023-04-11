FROM python:3.10
LABEL org.opencontainers.image.source=https://github.com/LLukas22/Retrieval-Augmented-QA
LABEL org.opencontainers.image.description="API hosting a Retrieval-Augmented-QA-Pipeline and Chat models on a CPU"
LABEL org.opencontainers.image.licenses=MIT


RUN apt-get update -y
RUN apt-get install -y python3-pip graphviz-dev git gcc-4.9 
RUN apt-get upgrade -y libstdc++6 

#Expose the ports
EXPOSE 8001

# Copy Files
RUN mkdir -p /app
ADD ./ /app/
WORKDIR /app

# Install Requirements
RUN --mount=type=cache,target=/root/.cache/pip pip3 install -r ./api/requirements-cpu.txt

RUN --mount=type=cache,target=/root/.cache/pip pip3 install git+https://github.com/huggingface/transformers@a17841ac4945631e4e13c072fa2a329b98ebb8b6 

ENV PYTHONPATH "${PYTHONPATH}:/app"

#Build cache dir for transformers
ENV HF_HOME "/huggingface/cache"

CMD ["python3", "/app/api/main.py"]