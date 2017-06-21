FROM tensorflow/tensorflow
RUN pip install boto3 && mkdir /tmp/imagenet && curl http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz | tar -C /tmp/imagenet -zxv
WORKDIR /root
COPY classify_image.py .