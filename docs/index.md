## PluGeN: Multi-Label Conditional Generation From Pre-Trained Models

[Maciej Wołczyk](https://scholar.google.com/citations?user=f6Xi7aoAAAAJ), [Magdalena Proszewska](https://scholar.google.com/citations?user=f6Xi7aoAAAAJ), [Łukasz Maziarka](https://scholar.google.com/citations?user=f6Xi7aoAAAAJ), [Maciej Zięba](https://scholar.google.com/citations?user=f6Xi7aoAAAAJ), [Patryk Wielopolski](https://scholar.google.com/citations?user=f6Xi7aoAAAAJ), [Rafał Kurczab](), [Marek Śmieja](https://scholar.google.com/citations?user=f6Xi7aoAAAAJ)

### Abstract 
> Modern generative models achieve excellent quality in a variety of tasks including image or text generation and chemical molecule modeling. However, existing methods often lack the essential ability to generate examples with requested properties, such as the age of the person in the photo or the weight of the generated molecule. Incorporating such additional conditioning factors would require rebuilding the entire architecture and optimizing the parameters from scratch. Moreover, it is difficult to disentangle selected attributes so that to perform edits of only one attribute while leaving the others unchanged. To overcome these limitations we propose PluGeN (Plugin Generative Network), a simple yet effective generative technique that can be used as a plugin to pre-trained generative models. The idea behind our approach is to transform the entangled latent representation using a flow-based module into a multi-dimensional space where the values of each attribute are modeled as an independent one-dimensional distribution. In consequence, PluGeN can generate new samples with desired attributes as well as manipulate labeled attributes of existing examples. Due to the disentangling of the latent representation, we are even able to generate samples with rare or unseen combinations of attributes in the dataset, such as a young person with gray hair, men with make-up, or women with beards. We combined PluGeN with GAN and VAE models and applied it to conditional generation and manipulation of images and chemical molecule modeling. Experiments demonstrate that PluGeN preserves the quality of backbone models while adding the ability to control the values of labeled attributes.

### Method

<table><tr>
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
    <em style="color: grey">(b) Probability distribution covered by PluGeN.</em>
  </p> 
</td>
</tr></table>

PluGeN factorizes true data distribution into components (marginal distributions) related to labeled attributes, see
(a), and allows for describing unexplored regions of data (uncommon combinations of labels) by sampling from independent
components, see (b). In the case illustrated here, PluGeN constructs pictures of men with make-up or women with beards,
although such examples rarely (or never) appear in the training set.

[Architecture](assets/img/schemat5.png)

Schema ...


