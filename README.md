# S2Dnet
Specular-to-Diffuse Translation for Multi-View Reconstruction.

## Paper
[Specular-to-Diffuse Translation for Multi-View Reconstruction](https://www.dropbox.com/s/ouvm6hh7ge5zdwy/S2Dnet.pdf?dl=0) 
ECCV, 2018 

<br/>

## Dependencies
* [Python](https://www.continuum.io/downloads)
* [PyTorch](http://pytorch.org/)
* [TensorFlow](https://www.tensorflow.org/) (optional for tensorboard)


<br/>

## Usage

### 1. Cloning the repository


### 2. Downloading the dataset

### 3. Training



### 4. Testing

```
### 5. Pretrained model
To download a pretrained model checkpoint, run the script below. The pretrained model checkpoint will be downloaded and saved into `./stargan_celeba_256/models` directory.

```bash
$ bash download.sh pretrained-celeba-256x256
```

To translate images using the pretrained model, run the evaluation script below. The translated images will be saved into `./stargan_celeba_256/results` directory.

```bash
$ python main.py --mode test --dataset CelebA --image_size 256 --c_dim 5 \
                 --selected_attrs Black_Hair Blond_Hair Brown_Hair Male Young \
                 --model_save_dir='stargan_celeba_256/models' \
                 --result_dir='stargan_celeba_256/results'
```

<br/>
