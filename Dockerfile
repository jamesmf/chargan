FROM tensorflow/tensorflow:2.3.1-gpu

WORKDIR /app

COPY . /app/chargan

RUN pip install -e chargan/

RUN python -m chargan.utils.prime

CMD /bin/bash