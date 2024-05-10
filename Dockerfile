FROM python:3.9.17-buster as builder

COPY ./requirements.txt /requirements.txt

RUN pip install -r /requirements.txt

FROM python:3.9-slim

ENV TZ='Asia/Shanghai' PYTHONUNBUFFERED='1' PYTHONIOENCODING='utf-8'

COPY --from=builder /usr/local/bin/uvicorn /usr/local/bin/uvicorn
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages

ENV PYTHONPATH=/usr/local/lib/python3.9/site-packages:/app

RUN python -c "import tiktoken;print(tiktoken.encoding_for_model('gpt-3.5-turbo').encode('1'));\
    print(tiktoken.get_encoding('cl100k_base').decode([16]))"

WORKDIR /app
COPY app app

ENTRYPOINT ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
