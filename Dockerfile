FROM python:3.8-slim

COPY Server/src/requirements.txt /root/Server/src/requirements.txt

RUN chown -R root:root /root/Server

WORKDIR /root/Server/src
RUN pip3 install -r requirements.txt

COPY Server/src/ ./
RUN chown -R root:root ./

ENV SECRET_KEY hello
ENV FLASK_APP run.py

RUN chmod +x run.py
CMD ["python3", "run.py"]
