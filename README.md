# foodai
Description

## Installion instructions
First build the docker image, type these command in the base directory:
```bash
docker build -t foodai .
```
And run the docker image:
```bash
docker run -it -p 8888:8888 -v $PWD/foodai:/notebooks/foodai foodai bash

```

## Docker cheatsheet
Remove all images and containers
```bash
#!/bin/bash
# Delete all containers
docker rm $(docker ps -a -q)
# Delete all images
docker rm $(docker images -q)
```

While using data volume containers, you have to remove container with -v flag as docker rm -v. If you don't use the -v flag, the volume will end up as a dangling volume and remain in to the local disk

To delete all dangling volumes, use the following command
```bash
docker volume rm `docker volume ls -q -f dangling=true`
```

## Datasets
- [Food-5k and Food-11](https://mmspg.epfl.ch/food-image-datasets)

## Models
[VGG16 model for Keras](https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3)

## Further notes
- [Very good instruction in Deep Learning with Keras and Tensorflow
](https://github.com/leriomaggio/deep-learning-keras-tensorflow)
- [All in one docker image for deep learning](https://github.com/floydhub/dl-docker)
- [Keras](https://github.com/keras-team/keras)
- [Tensorflow](https://github.com/tensorflow/tensorflow)
- [This Dockerfile is based on the Tensorflow image](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/docker)
