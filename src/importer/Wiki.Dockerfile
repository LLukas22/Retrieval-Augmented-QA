FROM python:3.10
LABEL org.opencontainers.image.source=https://github.com/LLukas22/Retrieval-Augmented-QA
LABEL org.opencontainers.image.description="Wikipedia importer for use with the Retrieval-Augmented-QA-API"
LABEL org.opencontainers.image.licenses=MIT

# Copy Files
RUN mkdir -p /app
ADD ./ /app/
WORKDIR /app

# Install Requirements
RUN --mount=type=cache,target=/root/.cache/pip pip3 install -r ./wikipedia/requirements.txt

ENV PYTHONPATH "${PYTHONPATH}:/app"

CMD ["python3", "/app/wikipedia/main.py"]