# Evo2HiC

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17917912.svg)](https://doi.org/10.5281/zenodo.17917912)

Evo2HiC is a multimodal and lightweight foundation model  for jointly modeling genomic sequences and structures.

## Introduction
<details>
<summary>Evo2HiC is a multimodal and lightweight foundation model  for jointly modeling genomic sequences and structures.</summary>

Understanding how genomic sequences shape three-dimensional (3D) genome architecture is fundamental to interpreting diverse biological processes. Although previous studies have demonstrated that sequence information can predict chromatin organization, they are unable to capture cell-type specific structures due to the lack of consideration of Hi-C datasets, which are widely available and capture rich structural information across biosamples.
Recently, DNA foundation models have demonstrated encouraging performance in capturing long-range genomic dependencies, holding promise for modeling chromatin interactions. However, the extremely high computational cost of running these models limits their applicability to Hi-C analysis, which requires genome-wide sequence embeddings.
Here, we present Evo2HiC, a lightweight foundation model for jointly modeling genomic sequences and structures. The key idea of Evo2HiC is to distill a large-scale DNA foundation model, Evo~2 (7B), into a compact encoder, while guiding the distillation with Hi-C data to preserve genomic features critical for 3D genome analysis.
When predicting Hi-C contact matrix using genome sequence, Evo2HiC improved Spearman correlation by 10.9\% over Orca.
Moreover, Evo2HiC achieved the best overall Pearson correlation when predicting five representative epigenomic assays by jointly embedding Hi-C and sequence information. Interpretation analysis of Evo2HiC revealed its ability to identify cell type–specific sequence motifs that explain changes in epigenomic signals. Finally, we demonstrated the cross-species generalizability of Evo2HiC on 177 species from the DNA Zoo dataset for Hi-C resolution enhancement.
In summary, Evo2HiC is a multimodal foundation model that integrates genome sequence and 3D chromatin structure information, substantially reduces computational cost while maintaining state-of-the-art accuracy across multiple chromatin analysis tasks, enables the identification of cell type-specific motifs, and demonstrates robust generalizability across species.

</details>

## Prerequisites
1. Git clone this repo
```bash
git clone https://github.com/bch9248/Evo2HiC.git
cd Evo2HiC
```
2. Installing miniconda with following commands:
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```
For more details about miniconda, check [the official document](https://www.anaconda.com/docs/getting-started/miniconda/main).

3. Install and activate required packages:
```bash
conda env create -f Evo2HiC.yaml -n [ENVNAME]
conda activate [ENVNAME]
```

4. add current path to python path:
```bash
export PYTHONPATH=($pwd):$PYTHONPATH
```

## Pretrained Models

Download pretrained Evo2HiC models from Zenodo:
https://doi.org/10.5281/zenodo.17917912

```bash
wget -c -O evo2hic_checkpoints.tar.zst "https://zenodo.org/records/17917912/files/evo2hic_checkpoints.tar.zst?download=1"
unzstd evo2hic_checkpoints.tar.zst
tar -xvf evo2hic_checkpoints.tar
```

### Task 1 Fine-tuning
1. Create account and get secret key on 4DN official website (https://data.4dnucleome.org/). Store the key ID and key in the keypairs.json.

2. Collect task 1 fine-tuning hic data (4DNFI2TK7L2F) from 4DN:

```bash
python collect_hic.py
```
3. Prepare genome sequence data:
```bash
mkdir -p data/hic/property
python download_hg38.py
python download_hg38_mappability.py
```

4. Run finetuning:
```bash
accelerate launch train/train_CDNA2d_Seq2HiC.py --initialize checkpoints/pretrained_weights/model.pt
```

### Inference finetuned models

1. Predicting Hi-C maps with finetuned model (Seq2HiC or ResEnh):

```bash
python inference/inference_CDNA2D.py -ckpt your_pretrained_ckpt --options_about_data
```
