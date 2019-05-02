# edgel3

Look, Listen, and Learn (L3) [3],  a  recently  proposed  state-of-the-art  transfer learning technique, mitigates the first challenge by training self-supervised deep audio embedding through binary Audio-Visual Correspondence,  and  the  resulting  embedding  can  beused to train a variety of downstream audio classification tasks. However, with close to 4.7 million parameters, the multi-layerL3-Net  CNN is still prohibitively expensive to be run on small edge devices, such as 'motes' that use a single microcontroller and limited memory to achieve long-lived self-powered operation. 

In EdgeL3 [1], we comprehensively explored the feasibility of compressing the L3-Net for mote-scale inference. We used pruning, ablation, and knowledge distillation techniques to show that the originally proposed L3-Net architecture is substantially overparameterized, not  only for AVC but for the target task of sound classification as evaluated on two popular downstream datasets, US8K and ESC50. EdgeL3, a 95% sparsified version of L3-Net, provides a useful reference model for approximating L3 audio embedding for transfer learning.

EdgeL3 is an open-source Python library for downloading the sparsified L3 models and computing deep audio embeddings from such models. The audio embedding models provided here are after sparsification and fine-tuning of L3 audio network. For additional implementation details, please refer to EdgeL3 [1]. The code for the model and training implementation can be found in  [https://github.com/ksangeeta2429/l3embedding/tree/dcompression](https://github.com/ksangeeta2429/l3embedding/tree/dcompression)

For non-sparse models and embedding, please refer to [OpenL3](https://github.com/marl/openl3) [2]

# References

Please cite the following papers when using EdgeL3 in your work:

[1] EdgeL3: Compressing L3-Net for Mote-Scale Urban Noise Monitoring <br/>
Sangeeta Kumari, Dhrubojyoti Roy, Mark Cartwright, Juan Pablo Bello, and Anish Arora. </br>
Parallel AI and Systems for the Edge (PAISE), Rio de Janeiro, Brazil, May 2019.

[2] Look, Listen and Learn More: Design Choices for Deep Audio Embeddings <br/>
Jason Cramer, Ho-Hsiang Wu, Justin Salamon, and Juan Pablo Bello.<br/>
IEEE Int. Conf. on Acoustics, Speech and Signal Processing (ICASSP), pages 3852–3856, Brighton, UK, May 2019.

[3] Look, Listen and Learn<br/>
Relja Arandjelović and Andrew Zisserman<br/>
IEEE International Conference on Computer Vision (ICCV), Venice, Italy, Oct. 2017.
