# benchmark-pytorch
This is a repository for testing Multi-GPU environments.

## How to run on a single GPU.
```:bash
$ cd cifar10
$ python3 single_gpu.py
```

## How to run distributed training on multi GPUs.

1. Login to the management node as user.
```
$ ssh user@host
```

2. Clone repository.
```
user@gpu-node:~/workspace$ git clone https://github.com/aizawan/benchmark-pytorch.git
```

3. Create job scripts of SBATCH. (You need to pull a pytorch image from nvidia gpu cloud in advance.)  
  - Sample script when using all of the available GPUs.
  ```
  #!/bin/bash
  #SBATCH -p defq
  #SBATCH -n 1
  #SBATCH -J benchmark-cifar10
  #SBATCH -o logs/stdout.%J
  #SBATCH -e logs/stderr.%J
  docker run --rm --runtime=nvidia --shm-size=32G -v /home/aizawa/workspace:/workspace nvcr.io/nvidia/pytorch:19.10-py3 python /workspace/benchmark-pytorch/cifar10/multi_gpu.py
  ```

  - Sample script when using the manually specified GPUs.
  ```
  #!/bin/bash
  #SBATCH -p defq
  #SBATCH -n 1
  #SBATCH -J benchmark-cifar10
  #SBATCH -o logs/stdout.%J
  #SBATCH -e logs/stderr.%J
  docker run --rm --runtime=nvidia -e CUDA_VISIBLE_DEVICES=0,1,3,6,9 --shm-size=32G -v /home/aizawa/workspace:/workspace nvcr.io/nvidia/pytorch:19.10-py3 python /workspace/benchmark-pytorch/cifar10/multi_gpu.py
  ```
