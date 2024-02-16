FROM tensorflow/tensorflow:latest-gpu
LABEL authors="bjoern"

WORKDIR /code

COPY ./requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

COPY ./src ./src
COPY ./src/data ./src/data

CMD ["python","app.py"]

#ENTRYPOINT ["top", "-b"]