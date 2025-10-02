# Multi-Label Conditional Generation From Pre-Trained Models

This repository contains the official code for PluGeN, a method for adapting pre-trained models for conditional multi-label generation. The paper was published in IEEE Transactions on Pattern Analysis and Machine Intelligence (shorter version appeared at AAAI 2022).

[[Full TPAMI Paper](https://ieeexplore.ieee.org/abstract/document/10480286/)] [[Short AAAI version](https://arxiv.org/abs/2109.09011)] [[Website](https://gmum.github.io/plugen)]

![image](./docs/assets/img/attributes_change.png)


## Abstract
Although modern generative models achieve excellent quality in a variety of tasks, they often lack the essential ability to generate examples with requested properties, such as the age of the person in the photo or the weight of the generated molecule. To overcome these limitations we propose PluGeN (Plugin Generative Network), a simple yet effective generative technique that can be used as a plugin for pre-trained generative models. The idea behind our approach is to transform the entangled latent representation using a flow-based module into a multi-dimensional space where the values of each attribute are modeled as an independent one-dimensional distribution. In consequence, PluGeN can generate new samples with desired attributes as well as manipulate labeled attributes of existing examples. Due to the disentangling of the latent representation, we are even able to generate samples with rare or unseen combinations of attributes in the dataset, such as a young person with gray hair, men with make-up, or women with beards. In contrast to competitive approaches, PluGeN can be trained on partially labeled data. We combined PluGeN with GAN and VAE models and applied it to conditional generation and manipulation of images, chemical molecule modeling and 3D point clouds generation.


## Code
The code is divided into three directories containing code for three different types of experiments:
* `CharVAE/` – experiments on the ZINC 250k dataset using the CharVAE backbone.
* `StyleGAN/` – experiments on the FFHQ dataset using the StyleGAN backbone. 
* `VAE/` – experiments on the CelebA dataset using the VAE backbone


## Acknowledgments

The research of M. Wołczyk was supported by the Foundation for Polish Science co-financed by the European Union under the European Regional Development Fund in the POIR.04.04.00-00-14DE/18-00 project carried out within the Team-Net program. The work carried out by Maciej Zieba was supported by the National Centre of Science (Poland) Grant No. 2020/37/B/ST6/03463. The work of Ł. Maziarka was supported by the National Science Centre (Poland) grant no. 2019/35/N/ST6/02125. The research of M. Śmieja was funded by the National Science Centre (Poland) grant no. 2022/45/B/ST6/01117.

## Citation
```
@article{proszewska2024multi,
  title={Multi-label conditional generation from pre-trained models},
  author={Proszewska, Magdalena and Wo{\l}czyk, Maciej and Zieba, Maciej and Wielopolski, Patryk and Maziarka, {\L}ukasz and {\'S}mieja, Marek},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume={46},
  number={9},
  pages={6185--6198},
  year={2024},
  publisher={IEEE}
}

@inproceedings{wolczyk2022plugen,
  title={Plugen: Multi-label conditional generation from pre-trained models},
  author={Wo{\l}czyk, Maciej and Proszewska, Magdalena and Maziarka, {\L}ukasz and Zieba, Maciej and Wielopolski, Patryk and Kurczab, Rafa{\l} and Smieja, Marek},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={36},
  number={8},
  pages={8647--8656},
  year={2022}
}
```
