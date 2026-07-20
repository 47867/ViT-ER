# ViT-ER
Visual Transformer for Emotion Recognition (4-class) <br><br>

<strong>Model checkpoints now available as .safetensors files over on Huggingface:</strong> <br>
SWIN-Base: [https://huggingface.co/Simon14142/VIT-ER_SWIN ](https://huggingface.co/Simon14142/emotion-recognition-4class-swin) <br>
DeiT-Base: [https://huggingface.co/Simon14142/VIT-ER_DeiT ](https://huggingface.co/Simon14142/emotion-recognition-4class-deit)<br>
<br>
Simplified inference script also available on Huggingface. <br>

## <strong>Introduction</strong> <br>
We introduce two machine learning models for visual emotion recognition. Using the CAER-S dataset by Lee et al. (2019) we predict four emotions: <br>
- Optimistic (happy)
- Pessimistic (sad + fear)
- Hostile (anger + disgust)
- Neutral (neutral)
<br><br>

We apply transfer learning on two Visual Transformers: <br>
- DeiT-Base-patch16-224 (86M) by Touvron et al. (2021) at Meta, pre-trained on ImageNet-1k
- SWIN-Base-patch4-window7-224 (86M) by Liu et al. (2021) at Microsoft, pre-trained on ImageNet-1k

<br><br>

We use a stratified 80/20 split for test and validation datasets. <br>
Images are unevenly distributed in our dataset, as our four emotion classes sometimes consist of two separate emotions. To account for that, we employ weighting. 
<br><br>

By using a mixed-precision approach, we can reduce compute cost while at the same time keep precision high where it matters most. <br>
Weights for the Random Sampler are computed at FP64 precision. All model weights are stored in FP32 precision. Gradient scaling as well as optimizer steps are also conducted with FP32 precision. Only forward passing and loss computation run in the lower FP16 precision to reduce compute cost. 
<br><br>

## <strong>Transfer learning</strong> <br>
Training is conducted for 35 epochs using a sophisticated multi-phase setup. <br>
We first freeze the backbone and train only the head for three epochs. Afterwards, we enable Mix-Up for two epochs. After five epochs, the backbone is unfrozen. Mix-up is disabled for the frist unfrozen epoch and gets enabled afterwards for all remaining epochs. <br>
After each epoch, validation loss is computed. A new checkpoint is saved when the validation loss reaches a new minimum. Should there be no improvement on the validation loss for seven epochs, training is stopped early. <br>
In our example, training runs for the full 35 epochs. The lowest validation loss is reached after 32 epochs.
<br><br>

Both models reach near-state-of-the-art performance with a Macro-F1 value >0.8 and validation accuracy of >0.8 on validation data. Single class performance is as follows: 
<br><br>

### <strong>SWIN-Base</strong>
| Class        | Precision | Recall | F1    |
|:-------------|:---------:|:------:|:-----:|
| Optimistic   | 0.820     | 0.845  | 0.832 |
| Pessimistic  | 0.925     | 0.869  | 0.896 |
| Hostile      | 0.882     | 0.853  | 0.867 |
| Neutral      | 0.671     | 0.778  | 0.721 |

Macro-F1: 0.8291, validation accuracy: 0.8442, validation loss: 0.4292 (after 32 epochs)
<br><br>

### <strong>DeiT-Base</strong>
| Class        | Precision | Recall | F1    |
|:-------------|:---------:|:------:|:-----:|
| Optimistic   | 0.795     | 0.810  | 0.802 |
| Pessimistic  | 0.906     | 0.864  | 0.884 |
| Hostile      | 0.882     | 0.802  | 0.840 |
| Neutral      | 0.612     | 0.776  | 0.689 |

Macro-F1: 0.8040, validation accuracy: 0.8198, validation loss: 0.4919 (after 32 epochs)
<br><br>
SWIN-Base consistently outperforms DeiT-Base, albeit by only a few points. 
<br><br>
## <strong>Domain adaptation</strong> <br>
We are conducting research on the KLIMAMEMES dataset by Haim et al. (2026). To align our model predictions better with this complicated dataset, we conduct domain adaptation. <br>
We manually annotate 410 images and conduct training for 80 epochs. For SWIN-Base we reach a Macro-F1 of 0.76 and Cohen's Kappa of 0.67. <br>
DeiT-Base is significantly weaker. Macro-F1 reaches 0.60 and Cohen's Kappa reaches 0.45. Improving on these results appears non-trivial.
<br><br>

## <strong>Hardware information</strong> <br>
Both models are trained on a single NVIDIA Blackwell GB203 GPU with 16 GB of VRAM (GDDR7, non-ECC, 28 Gigabit/s over 256bit-bus = 896 GiB/s transfer speed). The GPU is connected to the system via PCI-Express Gen. 5.0 with 16 lanes. CPU-side is handled by a 12th Gen. (Alder Lake) Intel Core i5-12600K on a motherboard with a Z690 chipset.  Our system has access to 64 GB of DDR4-DRAM (non-ECC) running with JEDEC specification for 3.200 MT/s at CL16-20-20-38 at 1.35V. We use a 1TB (~ 928.2 GiB) SAMSUNG 870QVO SSD as mass storage wich is connected via SATA-III (SATA-600).
<br><br>

At batch-size of 64, training takes around 1 hour for DeiT-Base (approx. 7 batches per second) and 1.5 hours for SWIN-Base (approx. 5.2 batches per second). DeiT-Base needs approx. 6.0 GiB of VRAM while SWIN-Base needs approx. 11.4 GiB of VRAM. Training should therefore be possible on a GPU with 12 GiB of VRAM. We still recommend a GPU with at least 16 GiB of VRAM. DRAM usage stays below 10 GiB for both models. Therefore, 16 GiB of DRAM should suffice, but 24 GiB or more are recommended.
<br><br>

At batch-size of 64 our Blackwell GB203 accelerator inferences around 400 images per second with single-precision (FP32) or around 800 images per second with half precision (BF16).
<br><br>

Our hardware setup shows that this task can be completed on mid-range consumer hardware, making it more accessible to researchers with a smaller budget and no access to datacenter-class hardware. 
<br><br>

This allows us to perform both training and inference entirely on local hardware. Data privacy and 100% control over all processes are thus guaranteed at all times. <br>
All electrical power used for training and inference stems from renewable energy sources, keeping environmental impact as low as possible.
<br><br>

## <strong>Software information</strong> <br>
We use Linux Fedora 43 for Workstation. All code runs in JupyterLab within a conda environment with Python 3.13.3. We use `torch` v2.11.0 with NVIDIA CUDA 12.8. <br>
All of the above mentioned software - except NVIDIA CUDA - is open-source and can be downloaded completely free of charge, making it accessible for everyone. 
<br><br>

## <strong>Publication</strong><br>
We publish the following data: <br>
- Code for CAER-S dataset pre-processing
- Code for both DeiT-Base and SWIN-Base model for transfer learning
- Code for checkpoint conversion from .pth to .safetensor
- Code for the selection of the dataset (images with human faces only)
- Code for model domain adaptation pre-processing
- Code for model domain adaptation training
- Code for model inference
- Domain adaptation annotations and sample weights
- Inference results
<br><br>

The model checkpoints are not published here, as they are too large to be uploaded here. But you can get them on Huggingface (see URL's at the start of ReadMe). <br>
Both the original CAER-S trained checkpoints as well as the domain-adapted checkpoints are available in the .safetensor file format. We recommend using .safetensor over .pth as the latter can be a security risk.
<br><br>

## <strong>Licenses</strong> <br>
We publish our entire code under a MIT license. Note that this does not include any models or datasets. A copy of the license is distributed with the repository. <br>
The DeiT-Base-patch16-224 model is distributed under the Apache-2.0 license. A copy of the Apache-2.0 license is distributed with this repository. <br>
The SWIN-Base-patch4-window7-224 model is also distributed under the Apache-2.0 license. <br>
The CAER-S dataset is distributed for research purposes only. 
<br><br>

## <strong>References</strong><br>
Haim, M., Haßler, J., Lübke, S., Ozornina, N., Stengl, L., Geigl, M., Plank, B., Peng, S. L., Zhou, S., Ommer, B., Schusterbauer, J., Grabe, M. E., & Mohammad, S. (2026). KLIMAMEMES dataset. https://klimamemes.ifkw.lmu.de/index.php/about-the-project/. <br>
Lee, J., Kim, S., Kim, S., Park, J., & Sohn, K. (2019). Context aware emotion recognition networks. Proceedings of the IEEE/CVF International Conference on Computer Vision. <br>
Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., & Guo, B. (2021). Swin Transformer: Hierarchical Vision Transformer using Shifted Windows. arXiv. https://doi.org/10.48550/arXiv.2103.14030 <br>
Touvron, H., Cord, M., Douze, M., Massa, F., Sablayrolles, A., & Jegou, H. (2021). Training data-efficient image transformers & distillation through attention. arXiv. https://doi.org/10.48550/arXiv.2012.12877 <br>


