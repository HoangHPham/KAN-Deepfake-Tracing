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

### Experiment 2: Interpretability validation

### Experiment 3: Adaptability to deepfake detection

## Quick Start 🚀





