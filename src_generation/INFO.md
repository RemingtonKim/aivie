# ***aivie*** - Art Generation

## Instructions
1. `cd src_generation` - navigate to directory.
2. `python main.py` - generate image to `../results` from images in `../data_img/img`

## Running in Colab
1. Upload `aivie.ipynb` to Google Colab
2. Arrange real and fake training images in the following directory structure:
    ```
    data_img_A
    │
    └───A
        |   fake_1.jpg
        |   fake_2.jpg
        |   ...

    data_img_B
    │
    └───B
        |   real_1.jpg
        |   real_2.jpg
        |   ...
    ```
3. Send `data_img_A` and `data_img_B` to .zip folders. Then, upload to Colab
4. Run all cells in notebook, downloading the trained model files.

## How it Works
The art generation in *aivie* is done with a generative adversarial network (GAN) architecture called Cycle-GAN, which allows unpaired image to image translation. This architecture was created by Zhu et al., and their original paper can be found [here](https://arxiv.org/pdf/1703.10593.pdf). The official implementation by [junyanz](https://github.com/junyanz/CycleGAN) as well as a simpler implementation by [aitorzip](https://github.com/aitorzip/PyTorch-CycleGAN) inspired my implementation. I rewrote the entire architecture to familiarize myself with pytorch and make necessary modifications. Cycle-GAN worked in *aivie* by translating real images to the style of different artists. The datasets used can be found [here](https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/).

## Dependencies
The deep learning library used for *aivie* is `pytorch`. `tqdm` is used to show progress while training the generator and discriminator networks. `numpy` is used for various mathematical operations. 

## Acknowledgements
Again, is based on the [paper](https://arxiv.org/pdf/1703.10593.pdf) by Zhu et al. and inspired by [junyanz](https://github.com/junyanz/CycleGAN)'s and [aitorzip](https://github.com/aitorzip/PyTorch-CycleGAN)'s implementation. However, I rewrote everything to familiarize myself with the pytorch library and make modifications

## Additional Info
* [Cycle-GAN explanation by Computerphile](https://www.youtube.com/watch?v=T-lBMrjZ3_0)