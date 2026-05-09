# ViT-ER
Visual Transformer for Emotion Recognition (4-class) </br></br>

We introduce two machine learning models for visual emotion recognition. Using the CAER-S dataset by Jiyoung et al. (2019) we predict four emotions: </br>
- Optimistic (happy)
- Pessimistic (sad + fear)
- Hostile (anger + disgust)
- Neutral (neutral)

We apply transfer learning on two Visual Transformers: </br>
- DeiT-Base-patch16-224 (86M) by Touvron et al. (2021) at Meta, pre-trained on ImageNet-1k
- SWIN-Base-patch4-window7-224 (86M) by Liu et al. (2021) at Microsoft, pre-trained on ImageNet-1k

</br>

By using a mixed-precision approach, we can reduce compute cost while at the same time keep precision high where it matters most. </br>
Weights for the Random Sampler are computed at FP64 precision. All model weights are stored in FP32 precision. Gradient scaling as well as optimizer steps are also conducted with FP32 precision. Only forward passing and loss computation run in the lower FP16 precision to reduce compute cost. </br></br>

Both models reach a Macro-F1 value >0.8 and validation accuracy of >0.8. Single class performance is as follows: </br>
<strong>SWIN-Base:</strong>
| Class        | Precision | Recall | F1    |
|:-------------|:---------:|:------:|:-----:|
| Optimistic   | 0.820     | 0.845  | 0.832 |
| Pessimistic  | 0.925     | 0.869  | 0.896 |
| Hostile      | 0.882     | 0.853  | 0.867 |
| Neutral      | 0.671     | 0.778  | 0.721 |

Macro-F1: 0.8291, validation accuracy: 0.8442, validation loss: 0.4292 (after 32 epochs)
</br></br>

<strong>DeiT-Base:</strong>
| Class        | Precision | Recall | F1    |
|:-------------|:---------:|:------:|:-----:|
| Optimistic   | 0.795     | 0.810  | 0.802 |
| Pessimistic  | 0.906     | 0.864  | 0.884 |
| Hostile      | 0.882     | 0.802  | 0.840 |
| Neutral      | 0.612     | 0.776  | 0.689 |

Macro-F1: 0.8040, validation accuracy: 0.8198, validation loss: 0.4919 (after 32 epochs)
</br></br>

<strong>Hardware information:</strong> </br>
Both models are trained on a single NVIDIA Blackwell GB203 GPU with 16 GiB of VRAM (GDDR7, non-ECC, 28 Gigabit/s over 256bit-bus = 896 GiB/s transfer speed). CPU-side is handled by a 12th Gen. (Alder Lake) Intel Core i5-12600K supported by 64 GiB of DDR4-DRAM (non-ECC) running with JEDEC specification for 3.200 MT/s at CL16-20-20-38 at 1.35V. </br></br>
Our hardware setup shows that this task can be completed on mid-range consumer hardware, making it more accessible to researchers with a smaller budget and no access to datacenter-class hardware. </br>
This allows us to perform both training and inference entirely on local hardware. Data privacy and 100% control over all processes are thus guaranteed at all times. </br>
All electrical power used for training and inference stems from renewable energy sources, keeping environmental impact as low as possible.</br></br>

<strong>Software information:</strong> </br>
We use Linux Fedora 43 for Workstation. All code runs in JupyterLab within a conda environment with Python 3.13.3. We use `torch` v2.11.0 with NVIDIA CUDA 12.8. </br>
All of the above mentioned software - except NVIDIA CUDA - is open-source and can be downloaded completely free of charge, making it accessible for everyone. 
