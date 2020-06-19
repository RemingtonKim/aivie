# ***aivie*** - Art Generation

## Instructions
1. `cd src_generation` - navigate to directory.
2. `python main.py` - generate image to `../results/abstract` from images in `../data_img/img` after training.

## Data Used
*aivie* was trained using 400 images from this [flower dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html). The abstract art used for training is from [Pexels](https://pexels.com). The keyword `abstract art` was used to search, and 400 images were selected. A sample image from this [dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) is provided in `../data_img/img`. Put images into that folder before running `./main.py`, as the images in `../data_img/img` will be converted to an abstract art style.  


## Running in Colab
1. Upload `aivie.ipynb` to Google Colab
2. Arrange flower and abstract art training images in the following directory structure:
    ```
    data_img_A
    │
    └───A
        |   abstract_1.jpg
        |   abstract_2.jpg
        |   ...

    data_img_B
    │
    └───B
        |   flower_1.jpg
        |   flower_2.jpg
        |   ...
    ```
3. Send `data_img_A` and `data_img_B` to .zip folders. Then, upload to Colab
4. Run all cells in notebook, downloading the trained model files to `../trained_models`.

## How it Works
The art generation in *aivie* is done with a generative adversarial network (GAN) architecture called Cycle-GAN, which allows unpaired image to image translation. This architecture was created by Zhu et al., and their original paper can be found [here](https://arxiv.org/pdf/1703.10593.pdf). The official implementation by [junyanz](https://github.com/junyanz/CycleGAN) as well as a simpler implementation by [aitorzip](https://github.com/aitorzip/PyTorch-CycleGAN) inspired my implementation. I rewrote the entire architecture to familiarize myself with pytorch, make necessary modifications while adding functionality. Cycle-GAN worked in *aivie* by translating images of flowers into the style of abstract art. 

## Dependencies
The deep learning library used for *aivie* is `pytorch`. `tqdm` is used to show progress while training the generator and discriminator networks. `numpy` is used for various mathematical operations. 

## Acknowledgements
Again, Cycle-GAN is based on the [paper](https://arxiv.org/pdf/1703.10593.pdf) by Zhu et al. and inspired by [junyanz](https://github.com/junyanz/CycleGAN)'s and [aitorzip](https://github.com/aitorzip/PyTorch-CycleGAN)'s implementation. However, I rewrote everything to familiarize myself with the pytorch library, make modifications, and add functionality. 

## Additional Info
* [Cycle-GAN explanation by Computerphile](https://www.youtube.com/watch?v=T-lBMrjZ3_0)