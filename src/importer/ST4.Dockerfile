FROM python:3.10
LABEL org.opencontainers.image.source=https://github.com/LLukas22/Retrieval-Augmented-QA
LABEL org.opencontainers.image.description="Schema ST4 importer for use with the Retrieval-Augmented-QA-API"
LABEL org.opencontainers.image.licenses=MIT

# Copy Files
RUN mkdir -p /app
ADD ./ /app/
WORKDIR /app

# Install Requirements
RUN --mount=type=cache,target=/root/.cache/pip pip3 install -r ./schema_st4/requirements.txt

ENV PYTHONPATH "${PYTHONPATH}:/app"

CMD ["python3", "/app/schema_st4/main.py"]