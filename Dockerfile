FROM python:3.12

COPY requirements.txt       /app/requirements.txt
RUN apt-get update && \
    pip install -U pip && \
    pip install -r /app/requirements.txt && \
    pip install "fastapi[standard]" python-dotenv

COPY img2vec_pytorch        /app/img2vec_pytorch
COPY setup.cfg              /app/setup.cfg
COPY setup.py               /app/setup.py
COPY LICENSE.txt            /app/LICENSE.txt
COPY README.md              /app/README.md

RUN cd /app && \
    pip install . && \
    cd / && rm -rf /app

ENTRYPOINT [ "uvicorn", "img2vec_pytorch.serve:app" ]
HEALTHCHECK CMD [ "curl", "127.0.0.1:8000/api/healthcheck" ]