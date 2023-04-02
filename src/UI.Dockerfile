FROM python:3.10

# Copy Files
RUN mkdir -p /app
ADD ./ /app/
WORKDIR /app

#Expose the ports
EXPOSE 8501

# Install Requirements
RUN --mount=type=cache,target=/root/.cache/pip pip3 install -r ./ui/requirements.txt

ENV PYTHONPATH "${PYTHONPATH}:/app"

ENTRYPOINT ["streamlit", "run", "/app/ui/Welcome.py", "--server.port=8501", "--server.address=0.0.0.0"]