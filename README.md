# foodai
Description

## Installion instructions
First build the docker image:
```bash
docker build -t foodai .
```
And run the docker image:
```bash
docker run -it -p 8888:8888 -v src:/root/src foodai bash

```

## Further notes
- [Very good instruction in Deep Learning with Keras and Tensorflow
](https://github.com/leriomaggio/deep-learning-keras-tensorflow)
- [All in one docker image for deep learning](https://github.com/floydhub/dl-docker)
- [Keras](https://github.com/keras-team/keras)
- [Tensorflow](https://github.com/tensorflow/tensorflow)
- [This Dockerfile is based on the Tensorflow image](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/docker)
