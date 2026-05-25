# Cross-Domain Mammography Cancer Diagnosis using Domain Adversarial Neural Networks

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![Status](https://img.shields.io/badge/Status-Manuscript%20In%20Preparation-yellow)

## Note
The core source code and training pipelines for this project are currently hosted in a private repository for security/institutional compliance. This repository serves as the public documentation and research summary.

## Overview

This repository contains the codes and scripts utilized in this project of cross-domain mammography cancer diagnosis, done as
part of a Research Assistantship at SciWhyLab, School of Computational & Integrative Sciences,
Jawaharlal Nehru University, New Delhi, under the supervision of **Prof. Shandar Ahmad**.

The central challenge addressed is **batch effects** arising from mammographic datasets of different geographical
and institutional origins. We develop and evaluate an ensemble framework of **Domain Adversarial Neural Networks (DANNs)**
to achieve robust cross-domain generalisation across diverse patient populations and imaging modalities.

---

## Datasets

The pipeline was evaluated on **seven datasets** spanning:
- **Populations:** Asian (CMMD, VinDR mammo), South Asian (IBIA), European (MIAS and INbreast), and American (CBIS-DDSM and RSNA)
- **Modalities:** Legacy (Scanned) Film Mammography and Full-Field Digital Mammography (FFDM)

---

## Pipeline Structure

```
cross-domain-mammography-DANN/
│
├── 1_preprocessing/
│   ├── Data_harmonization.py            # Set a standard filename for all the datasets
│   ├── dicom_to_png_and_sort.py         # DCM/PGM → PNG conversion
│   ├── MIAS_sorting                     # To set a standard dir structure for the MIAS dataset
│   ├── MIAS_pgm_to_png.py               # MIAS images are in .pgm format
│   ├── otsu_cropping.py                 # Otsu-based background cropping
│   └── CLAHE4x8.py                      # CLAHE enhancement on cropped images
│
├── 2_feature_extraction/ 
│   ├── EfficientNetB0-feature-extraction.py    # EfficientNet CNN feature extractor; pretrained on ImageNet1K_V1
│   ├── Medsiglip-feature-extractor.py          # Google/MedSigLIP-448 feature extractor; Loaded with the help of AutoProcessor module from HuggingFace 
│   ├── Radiomics-feature-extractor.py          # Handcrafted quantitative features
│   └── h5_inspect.py                           # Use this script to view and inspect the structure of the HDF5 files generated in the previous steps 
│
├── 3_classification/
│   ├── DANN/
│   │   ├── EfficientNet-SingleSource-DANN-training.py  # Running this script will train the DANN model on the EfficientNetB0 features 
│   │   ├── EfficientNet-SingleSource-DANN-ensemble.py  # Run this script to generate inference on test data by creating an ensemble of EfficientNet models generated in the previous step.
│   │   ├── Medsiglip-SingleSource-DANN-training.py     # Running this script will train the DANN model on the Google/MedSigLip-448 features
│   │   ├── Medsiglip-SingleSource-DANN-ensemble.py     # Run this script to generate inference on test data by creating an ensemble of Google/MedSigLip-448 models generated in the previous step.
│   │   ├── Radiomics-SingleSource-DANN-training.py     # Running this script will train the DANN model on the Quantitative features extracted with the PyRadiomics package 
│   │   └── Radiomics-SingleSource-DANN-ensemble.py     # Run this script to generate inference on test data by creating an ensemble of Radiomics models generated in the previous step.
│   ├── ML_classifier/
│   │   ├── EfficientNet-ML-classifier.py               # Run this script to classify the EfficientNetB0 features using XGB and RF
│   │   ├── Medsiglip-ML-classifier.py                  # Run this script to classify the Google/MedSigLip-448 features using XGB and RF
│   │   └── Radiomics-ML-classifier.py                  # Run this script to classify the Radiomics features using XGB and RF
│   │ 
│   └── Simple_MLP_classifier/
│       ├── EfficientNet-feature-classifier-Simple-MLP.py       # Run this script to classify the EfficientNetB0 features using a custom 4 layer MLP
│       ├── Medsiglip-features-classifier-Simple-MLP.py         # Run this script to classify the Google/MedSigLip-448 features using a custom 4 layer MLP
│       └── Radiomics-features-classifier-Simple-MLP.py         # Run this script to classify the Radiomics features using a custom 4 layer MLP
│
├── mammo_env.yml                 # Conda environment file
├── required_packages.txt         # pip requirements
└── README.md
```

---

## Methodology

### 1. Preprocessing
Three cohorts were created from each dataset:
- **Raw:** Unprocessed PNG images
- **Otsu-cropped:** Background removed using Otsu thresholding
- **CLAHE-enhanced:** Contrast Limited Adaptive Histogram Equalisation applied to Otsu-cropped images

### 2. Feature Extraction
Three feature types were extracted from each preprocessing cohort:
- **EfficientNet features:** Deep CNN representations
- **Google/MedSigLIP-448 features:** Medical vision-language model embeddings
- **Quantitative features:** Handcrafted radiomic-style features

### 3. Classification
Three classification approaches were evaluated:
- Classical ML  (Random Forest, XGBoost)
- Simple MLP    (4 Layer MLP)
- **Domain Adversarial Neural Networks (DANNs)** - one training and one classification script per feature type

### Primary Metric
**AUC-ROC** was used as the primary performance metric across all experiments.

---

## Key Results
- Domain Adversarial Neural Network consistently outperformed classical ML (XGB and RF) and simple MLP baselines in cross-domain settings.
- Google/MedSigLIP-448 features yielded strongest domain-invariant representations for digitally acquired images.
- Quantitative features yielded best features for domain-invariant representions for legacy (scanned) images.
- CNN based features completely failed in providing any generalizability across any of the 7 datasets or any of the 3 classifier models.

---

## Environment Setup
It is recommended to create a separate conda environment using the command given below. This will create a new conda env named "mammo_env" with all the python packages used in this study.

Using conda:
```bash
conda env create -f mammo_env.yml
conda activate mammo_env
```
If it is preferred by you to use pip, then use this command to install all the packages using pip

Using pip:
```bash
pip install -r required_packages.txt
```

All of this work was done on a Linux server running on a x86_64 AMD 48 core processor and a NVIDIA GPU with 24 GB VRAM, having CUDA version 11.5.
The total number of images used in this study was 19,557 (38 GB). 

---

## Status
Manuscript in preparation. Repository and the quantitative results will be kept private till publication.

---

## Authors
**Ankesh Kumar Pandey**
Research Assistant, SciWhyLab, SC&IS, JNU, New Delhi
ankeshpandey67@gmail.com

**Principal Investigator: Prof. Shandar Ahmad**
School of Computational & Integrative Sciences, JNU
shandar@jnu.ac.in

---

## Funding
National Network Project for Research in Bioinformatics, Department of Biotechnology, Government of India.

---


## References & Acknowledgments

If you use this work, please cite the original authors of the core models and libraries utilized in this framework:
# Datasets
* **CBIS-DDSM:** Sawyer-Lee, R., Gimenez, F., Hoogi, A., & Rubin, D. (2016). Curated Breast Imaging Subset of Digital Database for Screening Mammography (CBIS-DDSM) [Data set]. The Cancer Imaging Archive. https://doi.org/10.7937/K9/TCIA.2016.7O02S9CY 
* **CMMD:** Cui, Chunyan; Li Li; Cai, Hongmin; Fan, Zhihao; Zhang, Ling; Dan, Tingting; Li, Jiao; Wang, Jinghua. (2021) The Chinese Mammography Database (CMMD): An online mammography database with biopsy confirmed types for machine diagnosis of breast. The Cancer Imaging Archive. DOI: https://doi.org/10.7937/tcia.eqde-4b16
* **MIAS:** Suckling, J., Parker, J., Dance, D., Astley, S., Hutt, I., Boggis, C., Ricketts, I., Stamatakis, E., Cerneaz, N., Kok, S., Taylor, P., Betal, D., &amp; Savage, J. (2015). Mammographic Image Analysis Society (MIAS) database v1.21. Apollo - University of Cambridge Repository. https://doi.org/10.17863/CAM.105113
* **RSNA:** Trivedi, H. M., Vazirabad, M., Kitamura, F. C., Chen, Y., & Frazer, H. (2026). Open-Source Dataset for the RSNA Screening Mammography Cancer Detection Challenge. Radiology: Artificial Intelligence, e250375.
* **VinDR mammo:** Nguyen, H. T., Nguyen, H. Q., Pham, H. H., Lam, K., Le, L. T., Dao, M., & Vu, V. (2023). VinDr-Mammo: A large-scale benchmark dataset for computer-aided diagnosis in full-field digital mammography. Scientific Data, 10(1), 277.
* **INbreast:** Moreira, I. C., Amaral, I., Domingues, I., Cardoso, A., Cardoso, M. J., & Cardoso, J. S. (2012). Inbreast: toward a full-field digital mammographic database. Academic radiology, 19(2), 236-248.
* **IBIA:** https://ibdc.dbt.gov.in/ibia/study_details_browse/MAMOS_1000000004/ 

# Models and Techniques
* **EfficientNet:** Tan, M., & Le, Q. (2019, May). Efficientnet: Rethinking model scaling for convolutional neural networks. In International conference on machine learning (pp. 6105-6114). PMLR.
* **MedSigLIP:** Sellergren, A., Kazemzadeh, S., Jaroensri, T., Kiraly, A., Traverse, M., Kohlberger, T., ... & Yang, L. (2025). Medgemma technical report. arXiv preprint arXiv:2507.05201.
* **Domain-Adversarial Neural Networks:** Ajakan, H., Germain, P., Larochelle, H., Laviolette, F., & Marchand, M. (2014). Domain-adversarial neural networks. arXiv preprint arXiv:1412.4446.
* **Domain-Adversarial training of Neural Networks:** Ganin, Y., Ustinova, E., Ajakan, H., Germain, P., Larochelle, H., Laviolette, F., ... & Lempitsky, V. (2016). Domain-adversarial training of neural networks. Journal of machine learning research, 17(59), 1-35.
* **Gradient Reversal Layer:** Ganin, Y., & Lempitsky, V. (2015, June). Unsupervised domain adaptation by backpropagation. In International conference on machine learning (pp. 1180-1189). PMLR.
* **Gated Attention based MIL pooling:** Ilse, M., Tomczak, J., & Welling, M. (2018, July). Attention-based deep multiple instance learning. In International conference on machine learning (pp. 2127-2136). PMLR.
* **XGBoost:** Chen, T., & Guestrin, C. (2016, August). Xgboost: A scalable tree boosting system. In Proceedings of the 22nd acm sigkdd international conference on knowledge discovery and data mining (pp. 785-794).
* **Random Forest:** Breiman, L. (2001). Random forests. Machine learning, 45(1), 5-32.
* **Otsu segmentation:** Otsu, N. (1975). A threshold selection method from gray-level histograms. Automatica, 11, 285-296.
* **CLAHE:** Zuiderveld, K. (1994). Contrast limited adaptive histogram equalization. In Graphics gems IV (pp. 474-485).

# Core Libraries
* **PyRadiomics:** van Griethuysen, J. J. M., Fedorov, A., Parmar, C., Hosny, A., Aucoin, N., Narayan, V., Beets-Tan, R. G. H., Fillion-Robin, J. C., Pieper, S., & Aerts, H. J. W. L. (2017). Computational Radiomics System to Decode the Radiographic Phenotype. Cancer research, 77(21), e104–e107. https://doi.org/10.1158/0008-5472.CAN-17-0339
* **PyTorch & Torchvision:** Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., ... & Chintala, S. (2019). Pytorch: An imperative style, high-performance deep learning library. Advances in neural information processing systems, 32.

## License
To be added upon manuscript submission.
