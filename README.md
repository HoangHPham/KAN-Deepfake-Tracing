<h2 align="center">Kolmogorov–Arnold Network-based Model for Interpretable and Adaptive Speech Deepfake Source Tracing</h2>

![proposed_model](assets/proposed_model.png)

## Updates

- 2025/11/27: The official implementation has been open-sourced.

## Table of Contents

- [Technical Briefing 💡](#)
- [Main Results 🏆](#)
  - [1. Source tracing tasks](#)
  - [2. Interpretability validation](#)
  - [3. Adaptability to deepfake detection](#)
- [Quick Start 🚀](#)

## Technical Briefing 💡

**Introducing proposed model** - the next generation of speech deepfake source tracing system with interpretability and adaptability.  

* **CM_MTL_KAN: A combination of countermeasure based Multi-task Learning and Kolmogorov-Arnold Network**

  * Two architectures of deepfake detection model, namely AASIST and SSL-AASIST, are used as backbones for audio representation learning.
  * Performs two main source tracing tasks: (1) attack attribute classification and (2) attack type classification.
  * Three training strategies are (1) training from scratch, (2) full finetuning and (3) partial finetuning.

* **Intepretability mechanism: A transparency using both Intrinsic and Extrinsic factors**

  * Intrinsic interpretability: an integration in KAN with the auxiliary structure obtained from metadata of ASVspoof2019-attr-17 protocol in ASVspoof 2019 LA dataset. 
  * Extrinsic interpretability: a KAN's built-in mechanism for input feature importance analysis. 
  * Feature importance results are validated using three criteria: (1) consistency, (2) stability and (3) faithfulness.
 
* **OOD detection: An adaptability from source tracing to deepfake detection**

  * Two kinds of embeddings are extracted: (1) attack attribute embeddings and (2) attack classification embeddings.
  * Evaluating how discriminative embeddings are between spoofed speech (In-distribution samples) and bonafide speech (Out-of-distribution samples).
  * Two algorithms are used: (1) OOD Detection with Deep Nearest Neighbors (distance-based) and (2) OOD Detection using indegree number in directed kNN graph (in-degree based)       

## Main Results 🏆

### Experiment 1: Source tracing performance
Table 1: Performance of our proposed source tracing model on ASVspoof2019-attr-17 protocol of ASVspoof 2019 LA dataset. 
![exp_res_proposed_model](assets/exp_res_proposed_model.png)

### Experiment 2: Interpretability validation

#### Intrinsic interpretability
A visualization of structured KAN module that presents the relations between attack attributes and attack types in ASVspoof 2019 LA dataset.
![visualization_SKM](assets/visualization_SKM.png)

#### Extrinsic interpretability
A bar plot shows the global importance scores of each attack attribute that affects the performance of attack type classification.

![global_FI_ranks_model_level](assets/global_FI_ranks_model_level.png)

### Experiment 3: Adaptability to deepfake detection
Table 2: Performance of OOD detection. 
![exp_res_ood_detection](assets/exp_res_ood_detection.png)

## Quick Start 🚀

### 1. Install Dependencies

```
git clone https://github.com/HoangHPham/KAN-Deepfake-Tracing.git
cd KAN-Deepfake-Tracing
conda create -n sourceTracing python=3.12
conda activate sourceTracing
pip install -r requirements.txt
```

### 2. Data preparation 

- Download **ASVspoof 2019 LA dataset** at [ASVspoof2019](https://datashare.ed.ac.uk/handle/10283/3336).
- Two data protocols, i.e., **ASVspoof2019-attr-2** and **ASVspoof2019-attr-17**, are available at `./data`.
- 🔥 Modify data paths in configurations files:

`./data/ASVspoof2019_LA_cm.yaml`
``` code
# ASVspoof2019-attr-2 protocol
train_dataset_path: <DATADIR>/asvspoof2019/LA/ASVspoof2019_LA_train
dev_dataset_path: <DATADIR>/asvspoof2019/LA/ASVspoof2019_LA_dev
eval_dataset_path: <DATADIR>/asvspoof2019/LA/ASVspoof2019_LA_eval 

train_protocol_path: ./data/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt
dev_protocol_path: ./data/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt
eval_protocol_path: ./data/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt
```

`./data/ASVspoof2019_attr17_cm.yaml`
``` code
# ASVspoof2019-attr-17 protocol
train_dataset_path: <DATADIR>/asvspoof2019/asvspoof2019_attr17/ASVspoof2019_attr17_train
dev_dataset_path: <DATADIR>/asvspoof2019/asvspoof2019_attr17/ASVspoof2019_attr17_dev
eval_dataset_path: <DATADIR>/asvspoof2019/asvspoof2019_attr17/ASVspoof2019_attr17_eval

train_protocol_path: ./data/ASVspoof2019_attr17_cm_protocols/Train_ASVspoof19_attr17.txt 
dev_protocol_path: ./data/ASVspoof2019_attr17_cm_protocols/Dev_ASVspoof19_attr17.txt
eval_protocol_path: ./data/ASVspoof2019_attr17_cm_protocols/Eval_ASVspoof19_attr17.txt 
```

- For data pre-processing, both silence trimming and RawBoost augmentation are used as default. See `./data_utils.py` for more details. 

### 3. Training

- 🔥 Modify hyper-parameters in `./config/train.yaml`. Some importances:

``` code
# model config
backbone: # AASIST or SSLAASIST
use_pretrained_backbone: # True if using pretrained weights for backbone
freeze_backbone: # True if freezing backbone
use_kan_auxiliary_structure: # True if using auxiliary structure for KAN module

# Resume pretrained weights
resume: # True if training from a pretrained weights
pretrained_ppm_path: # requiring a path for pretrained proposed model if resume=True

# data pre-processing
trim_silence: # True if trimming silence segments in input audio 
```

- 🔥 **Note:** if `use_pretrained_backbone=True`, pretrained weights of backbone are required. [AASIST](https://github.com/clovaai/aasist) or [SSL-AASIST](https://github.com/TakHemlata/SSL_Anti-spoofing) should be trained first for deepfake detection task to get the pretrained weights. 

- To run a training:
``` code
python train.py
```

### 4. Validation

- 🔥 Modify hyper-parameters in `./config/evaluate.yaml` (most are similar to training configurations):

``` code
# model config
...
pretrained_weights_path: <path to trained weights>

# ...
```

### 5. Interpretability

### 6. OOD detection

### 7. Additional experiments

### 7.1. Baseline models

### 7.2. Ablations

## Acknowledgements
Very thanks to authors who have foundations for my implementations:
1. [Official AASIST by @clovaai](https://github.com/clovaai/aasist)
2. [Official SSL-AASIST by @TakHemlata](https://github.com/TakHemlata/SSL_Anti-spoofing)
3. [Official KAN by @KindXiaoming](https://github.com/KindXiaoming/pykan)
4. [Baseline source tracing model by @Manasi2001](https://github.com/Manasi2001/Spoofed-Speech-Attribution)











