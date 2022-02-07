
<table><tr>
<td> 
  <p align="center" style="padding: 10px">
    <img alt="Forwarding" src="assets/img/attributes_change.png">
    <br>
    <em style="color: grey">Attributes manipulation performed by PluGeN using the StyleGAN backbone.</em>
  </p> 
</td>
</tr></table>

## Abstract 

> Modern generative models achieve excellent quality in a variety of tasks including image or text generation and chemical molecule modeling. However, existing methods often lack the essential ability to generate examples with requested properties, such as the age of the person in the photo or the weight of the generated molecule. Incorporating such additional conditioning factors would require rebuilding the entire architecture and optimizing the parameters from scratch. Moreover, it is difficult to disentangle selected attributes so that to perform edits of only one attribute while leaving the others unchanged. To overcome these limitations we propose PluGeN (Plugin Generative Network), a simple yet effective generative technique that can be used as a plugin to pre-trained generative models. The idea behind our approach is to transform the entangled latent representation using a flow-based module into a multi-dimensional space where the values of each attribute are modeled as an independent one-dimensional distribution. In consequence, PluGeN can generate new samples with desired attributes as well as manipulate labeled attributes of existing examples. Due to the disentangling of the latent representation, we are even able to generate samples with rare or unseen combinations of attributes in the dataset, such as a young person with gray hair, men with make-up, or women with beards. We combined PluGeN with GAN and VAE models and applied it to conditional generation and manipulation of images and chemical molecule modeling. Experiments demonstrate that PluGeN preserves the quality of backbone models while adding the ability to control the values of labeled attributes.

## Intuition

<table>
<tr>
<td> 
  <p align="center" style="padding: 10px">
    <img alt="Forwarding" src="assets/img/PluGEN_2D_9_mod3.png">
    <br>
    <em style="color: grey">(a) Factorization of true data distribution.</em>
  </p> 
</td>
<td> 
  <p align="center">
    <img alt="Routing" src="assets/img/PluGEN_2D_8_mod3.png">
    <br>
    <em style="color: grey">(b) Distribution covered by PluGeN.</em>
  </p> 
</td>
</tr>
<tr>  
    <td colspan="2">
    <p align="center">
        <em style="color: grey">PluGeN factorizes true data distribution into components (marginal distributions) related to labeled attributes, see (a), and allows for describing unexplored regions of data (uncommon combinations of labels) by sampling from independent components, see (b). In the case illustrated here, PluGeN constructs pictures of men with make-up or women with beards, although such examples rarely (or never) appear in the training set.</em>
    </p>
    </td>
</tr>
</table>

## Method

<table><tr>
<td> 
  <p align="center">
    <img alt="Routing" src="assets/img/schemat5.png">
    <br>
    <em style="color: grey">PluGeN maps the entangled latent space Z of pretrained generative models using invertible normalizing flow into a separate space, where labeled attributes are modeled using independent 1-dimensional distributions. By manipulating label variables in this space, we fully control the generation process.</em>
  </p> 
</td>
</tr></table>

## Results

### Attribute manipulation 

<table>
<tr>
<td width="33%"> 
  <p align="center">
    <img alt="Forwarding" src="assets/img/age.gif">
    <br>
    <em style="color: grey">Age</em>
  </p> 
</td>
<td width="33%"> 
  <p align="center">
    <img alt="Routing" src="assets/img/baldness.gif">
    <br>
    <em style="color: grey">Baldness</em>
  </p> 
</td>
<td width="33%"> 
  <p align="center">
    <img alt="Routing" src="assets/img/rotate.gif">
    <br>
    <em style="color: grey">Yaw</em>
  </p> 
</td>
</tr>
<tr>  
    <td colspan="3">
    <p align="center">
        <em style="color: grey">Gradual modification of attributes performed on the StyleGAN latent codes. </em>
    </p>
    </td>
</tr>
</table>

<br>

<table><tr>
<td> 
  <p align="center">
    <img alt="Routing" src="assets/img/vae-plugen-manipulation.png">
    <br>
    <em style="color: grey"> Examples of image attribute manipulation using VAE backbone.</em>
  </p> 
</td>
</tr></table>

### Conditional generation

<table><tr>
<td> 
  <p align="center">
    <img alt="Routing" src="assets/img/generation_without_frame.png">
    <br>
    <em style="color: grey"> Examples of conditional generation using VAE backbone. Each row contains the same person (style variables) with modified attributes (label variables).</em>
  </p> 
</td>
</tr></table>

### Chemical molecules modeling

<table>
<tr>
<td> 
  <p align="center" style="padding: 10px">
    <img alt="Forwarding" src="assets/img/traverse_mols_v2.png">
    <br>
    <em style="color: grey">(a) Molecules decoded from path.</em>
  </p> 
</td>
<td> 
  <p align="center">
    <img alt="Routing" src="assets/img/traverse_logP_v2.png">
    <br>
    <em style="color: grey">(b) LogP of presented molecules.</em>
  </p> 
</td>
</tr>
<tr>  
    <td colspan="2">
    <p align="center">
        <em style="color: grey"> Molecules obtained by the model during an optimization phase (a), and their LogP (b).</em>
    </p>
    </td>
</tr>
</table>

## Related works

  * Conditional VAE (cVAE)  (Kingma et al. 2014)
  * Conditional GAN (cGAN)  (Mirza and Osindero 2014; Perarnau et al. 2016; He et al. 2019)
  * Fader Networks  (Lample et al. 2017)
  * MSP  (Li et al. 2020)
  * CAGlow  (Liu et al. 2019) 
  * StyleFlow  (Abdal et al. 2021)
  * InterFaceGAN  (Shen et al. 2020)
  * Hijack-GAN  (Wang, Yu, and Fritz 2021)

## Acknowledgments

  * The research of M. Wołczyk was supported by the Foundation for Polish Science co-financed by the European Union under the European Regional Development Fund in the POIR.04.04.00-00-14DE/18-00 project carried out within the Team-Net programme.
  * The research of M. Proszewska was supported by the National Science Centre (Poland), grant no. 2018/31/B/ST6/00993.
  * The research of Ł. Maziarka was supported by the National Science Centre (Poland), grant no. 2019/35/N/ST6/02125. 
  * The research of M. Zieba was supported by the National Centre of Science (Poland), grant No. 2020/37/B/ST6/03463.
  * The research of R. Kurczab was supported by the Polish National Centre for Research and Development (Grant LIDER/37/0137/L9/17/NCBR/2018).
  * The research of M. Smieja was funded by the Priority Research Area DigiWorld under the program Excellence Initiative -– Research University at the Jagiellonian University in Krakow.

## References

  * Laurent Dinh, David Krueger, and Yoshua Bengio. Nice: Non-linear independent components estimation. arXiv preprint arXiv:1410.8516, 2014.
  * Rafael Gómez-Bombarelli, Jennifer N Wei, David Duvenaud, José Miguel Hernández-Lobato, Benjamín Sánchez-Lengeling, Dennis Sheberla,Jorge Aguilera-Iparraguirre, Timothy D Hirzel, Ryan P Adams, and Alán Aspuru-Guzik. Automatic chemical design using a data-driven continuous representation of molecules. ACS central science, 4(2):268–276, 2018.
  * Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, and Timo Aila. Analyzing and improving the image quality of StyleGAN. In Proc. CVPR, 2020.
  * Xiao Li, Chenghua Lin, Ruizhe Li, Chaozheng Wang, and Frank Guerin. Latent space factorisation and manipulationvia matrix subspace projection. In International Conference on Machine Learning, pages 5916–5926. PMLR, 2020.

## Bibtex
```bibtex
@misc{wołczyk2022plugen,
      title={{PluGeN: Multi-Label Conditional Generation From Pre-Trained Models}}, 
      author={Maciej Wołczyk and Magdalena Proszewska and Łukasz Maziarka and Maciej Zięba and Patryk Wielopolski and Rafał Kurczab and Marek Śmieja},
      year={2022},
      eprint={2109.09011},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```