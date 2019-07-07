# Emoji-GAN
[![License MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![HitCount](http://hits.dwyl.io/HitLuca/GAN-collection.svg)](http://hits.dwyl.io/HitLuca/GAN-collection)
[![Total alerts](https://img.shields.io/lgtm/alerts/g/HitLuca/GAN-collection.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/HitLuca/GAN-collection/alerts/)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/HitLuca/GAN-collection.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/HitLuca/GAN-collection/context:python)

## Description
Collection of various GAN models using my own way of implementing them.

### Models
At this point four different generative models are implemented:

* Conditional Improved Wasserstein GAN (CWGAN-GP)
* Improved Wasserstein GAN (WGAN-GP)
* Boundary Equilibrium GAN (BEGAN)
* Deep Convolutional GAN (DCGAN)
* Variational AutoEncoder with learned similarity metric (WGAN-GP-VAE)

The WGAN-GP-VAE model implementation comes from my [other](https://github.com/HitLuca/GANs_for_spiking_time_series) project, as a part of my Master Thesis (AI)

## Project structure
The [```models```](gan_collection/models) folder contains all the generative models implemented.

### Prerequisites
To install the python environment for this project, refer to the [Pipenv setup guide](https://pipenv.readthedocs.io/en/latest/basics/)
