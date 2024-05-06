FROM python:3.10.12
ENV PYTHONUNBUFFERED = 1

WORKDIR  /app
RUN apt-get update && apt-get install -y cmake
COPY requirements.txt .

RUN pip --no-cache-dir install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python3", "manage.py" ;"runserver","0.0.0.0:8000"]