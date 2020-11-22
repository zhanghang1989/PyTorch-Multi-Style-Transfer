#!/bin/bash
docker run --rm -it --gpus all -v $(pwd)/..:/style-transfer style-transfer
