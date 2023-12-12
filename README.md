# The Journey, Not the Destination: How Data Guides Diffusion Models

[![arXiv](https://img.shields.io/badge/arXiv-2311.06205-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2312.06205)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Check out our [blog post](https://gradientscience.org/diffusion-trak/)!

In our [paper](https://arxiv.org/abs/2312.06205), we introduce a data attribution framework for diffusion models, together with an efficent method fo computing attribution scores. Given a generated image `X` and a diffusion model of interest, you can use our library to identify training examples which strongly guide the diffusion model towards generating `X`.

In particular, we provide attribution scores for each *step* of the diffusuion process:
<div style="width:65%; margin: 0 auto;">
  <table style="width:100%;">
    <tr>
      <td><img src="assets/mscoco.gif" alt="MSCOCO Attributions GIF"></td>
      <td><img src="assets/cifar.gif" alt="CIFAR10 Attributions GIF"></td>
    </tr>
  </table>
</div>


# Usage
Check out the [examples](https://github.com/MadryLab/journey-TRAK/tree/main/examples). There, we:
- provide pre-computed attribution features so you can quickly score your generated images
- showcase how to compute the final scores using pre-computed features
- provide scripts to compute attribution features

Our code is based on the [TRAK API](https://github.com/MadryLab/trak).

# Citation
If you use this code in your work, please cite using the following BibTeX entry:

```
@inproceedings{georgiev2023journey,
      title={The Journey, Not the Destination: How Data Guides Diffusion Models}, 
      author={Kristian Georgiev and Joshua Vendrow and Hadi Salman and Sung Min Park and Aleksander Madry},
      booktitle = {Arxiv preprint arXiv:2312.06205},
      year={2023},
}
```
