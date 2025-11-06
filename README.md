# Evo2HiC
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

1. Installing miniconda with following commands:
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ~/Miniconda3-latest-Linux-x86_64.sh
```
For more details about miniconda, check [the official document](https://www.anaconda.com/docs/getting-started/miniconda/main).

2. Install and activate required packages:
```bash
conda env create -f Evo2HiC.yaml
conda activate Evo2HiC
```

3. add current path to python path:
```bash
export PYTHONPATH=($pwd):$PYTHONPATH
```

## Usage
### Pretraining

1. Collect hic data from ENCODE and 4DN (full index in 'data/hic_index.tsv'). Save the hic data to hic_data_dir specified in config.py.

2. Run this command to start pretraining:

```bash
accelerate launch train/pretrain.py
```

The training process will be recorded by weight-and-bias(wandb).
Read 'train/pretrain.py' if you want to change hyper-parameters.

### inference based on pretrained checkpoint

1. Calculate 1D DNA embeddings:

```bash
python inference/inference_dna_embeds.py -ckpt your_pretrained_ckpt
```

2. Calculate 2D DNA embeddings and Hi-C embeddings for retrieval:

```bash
python inference/retrieval_siglip.py -ckpt your_pretrained_ckpt
```

### Finetune pretrained checkpoint for predicting Hi-C contact matrix using genome sequences

1. Finetine the pretrained model:

```bash
accelerate launch train/train_CDNA2d_Seq2HiC.py --initialize your_pretrained_ckpt
```

### Finetune pretrained checkpoint for Hi-C resolution enhancement

1. Finetine the pretrained model:

```bash
accelerate launch train/train_CDNA2d_ResEnh.py --initialize your_pretrained_ckpt
```

### Finetune pretrained checkpoint for predicting epigenomic profiles

1. Finetine the pretrained model:

```bash
accelerate launch train/train/train_CDNA1d.py --initialize your_pretrained_ckpt
```



