FROM gcr.io/tensorflow/tensorflow

LABEL maintainer="Simon Geyer <study@simongeyer.com>"

ARG KERAS_VERSION=2.1.2

# Install some dependencies
RUN apt-get update && apt-get install -y \
    git \
    && \
    apt-get clean && \
	rm -rf /var/lib/apt/lists/*

RUN pip --no-cache-dir install git+git://github.com/fchollet/keras.git@${KERAS_VERSION}
