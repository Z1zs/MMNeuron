# MMNeuron: Discovering Neuron-Level Domain-Specific Interpretation in Multimodal Large Language Model

<div>
<div align="center">
    <a href='https://openreview.net/profile?id=~Jiahao_Huo2' target='_blank'>Jiahao Huo<sup>1,3</sup></a> 
    <a href='https://openreview.net/profile?id=~Yibo_Yan1' target='_blank'>Yibo Yan<sup>1,2</sup></a> 
    <a href='https://scholar.google.com/citations?user=d6B67dUAAAAJ&hl=en' target='_blank'>Boren Hu<sup>1,2</sup></a> 
    <a href='https://ait.hkust-gz.edu.cn/archives/3729' target='_blank'>Yutao Yue<sup>1,2</sup></a> 
    <a href='https://xuminghu.github.io/' target='_blank'>Xuming Hu<sup>✉,1,2</sup></a> 
</div>
<div>
<div align="center">
    <sup>1</sup>The Hong Kong University of Science and Technology (Guangzhou) <br>   
    <sup>2</sup>The Hong Kong University of Science and Technology   
    <sup>3</sup>Tongji University <br>  
    <sup>✉</sup> Corresponding Author
</div>

---

Official implementation of "[MMNeuron: Discovering Neuron-Level Domain-Specific Interpretation in Multimodal Large Language Model](https://arxiv.org/abs/2406.11193)".
Our codes are borrowed from [Tang](https://github.com/StevenTang1998)'s language specific neurons implementation [here](https://github.com/RUCAIBox/Language-Specific-Neurons) and [nrimsky](https://github.com/nrimsky)'s logit lens implementation [here](https://github.com/nrimsky/LM-exp/blob/main/intermediate_decoding/intermediate_decoding.py). Thanks a lot for their efforts!

## Updates

- **17 June, 2024** :Paper published in Arxiv.
- **17 June, 2024** : Code published.
- **20 Steptember, 2024** : Paper accepted by EMNLP main conference!

---

This repository contains the **official implementation** of the following paper:

> **MMNeuron: Discovering Neuron-Level Domain-Specific Interpretation in Multimodal Large Language Model** https://arxiv.org/abs/2406.11193
>
> **Abstract:** _Projecting visual features into word embedding space has become a significant fusion strategy adopted by Multimodal Large Language Models (MLLMs). However, its internal mechanisms have yet to be explored. Inspired by multilingual research, we identify domain-specific neurons in multimodal large language models. Specifically, we investigate the distribution of domain-specific neurons and the mechanism of how MLLMs process features from diverse domains. Furthermore, we propose a three-stage framework for language model modules in MLLMs when handling projected image features, and verify this hypothesis using logit lens. Extensive experiments indicate that while current MLLMs exhibit Visual Question Answering (VQA) capability, they may not fully utilize domain-specific information. Manipulating domain-specific neurons properly will result in a 10% change of accuracy at most, shedding light on the development of cross-domain, all-encompassing MLLMs in the future. Our code will be released upon paper notification._

## Todo

1. [X] Release the code.

## Get Start

- [Install](#install)
- [Dataset](#dataset-preparation)
- [Activation](#neuron-activation)
- [Identification](#domain-specific-neuron-identification)
- [Generation](#generation)
- [Logit Lens](#logit-lens)

## Install

```shell
conda create -n mmneuron python=3.10
conda activate mmneuron
pip install -r requirements.txt
```

## Dataset Preparation

Download the following datasets and put them in directory named "benchs".

#### LingoQA

Auto Driving Domain. You can get the dataset [here](https://github.com/wayveai/LingoQA).

#### VQAv2

Common Life. You can get the dataset [here](https://visualqa.org/download.html).

#### DocVQA

Document VQA. You can get the dataset [here](https://www.docvqa.org/datasets/docvqa) (or huggingface link [here](https://huggingface.co/datasets/pixparse/docvqa-single-page-questions)).

#### PMC-VQA

Medical VQA. You can get the dataset [here](https://www.docvqa.org/datasets/docvqa).

#### RS-VQA(HS)

Remote sensing VQA. You can get the dataset [here](https://zenodo.org/records/6344367). 

The final directory should look like:   
├── ad  
│   ├── images  
│   ├── images.zip  
│   ├── train.parquet  
│   └── val  
├── med  
│   ├── figures  
│   ├── images  
│   ├── test_2.csv  
│   ├── test_clean.csv  
│   ├── test.csv  
│   ├── train_2.csv  
│   └── train.csv  
├── rs  
│   ├── Data  
│   ├── USGSanswers.json  
│   ├── USGSimages.json     
│   ├── USGSquestions.json      
│   ├── USGS_split_test_answers.json      
│   ├── USGS_split_test_questions.json      
│   ├── USGS_split_val_answers.json      
│   ├── USGS_split_val_images.json      
│   └── USGS_split_val_questions.json     
└── vqav2     
&emsp;        ├── data     
&emsp;        ├── README.md  
&emsp;        ├── train2014  
&emsp;        ├── v2_mscoco_train2014_annotations.json    
&emsp;        ├── v2_mscoco_val2014_annotations.json     
&emsp;        ├── v2_mscoco_val2014_ansdict.json    
&emsp;        ├── v2_OpenEnded_mscoco_train2014_questions.json     
&emsp;        ├── v2_OpenEnded_mscoco_val2014_questions.json     
&emsp;        └── val2014     
   
## Neuron Activation

After preparing the data, you can record the activation probability of neurons in LLaVA-Next and InstructBLIP by running the python script:

```shell
python activation.py -m llava
python activation.py -m blip
```

To simplify, we use the [LLaVA-Next](https://huggingface.co/llava-hf/llava-v1.6-vicuna-7b-hf) and [InstructBLIP](https://huggingface.co/Salesforce/instructblip-vicuna-7b)'s huggingface API here, you can also find their official implementation in [URL1](https://github.com/haotian-liu/LLaVA/tree/main) and [URL2](https://github.com/salesforce/LAVIS/tree/main).

## Domain-specific Neuron Identification

After getting the activation probability, you can identifying the domain-specific neurons in LLaVA-Next or InstructBLIP's language model by running the python script:

```shell
python identify.py -m llava -d lang
```

For LLaVA-Next, the options of '-d' are ['lang','vision','mmproj'].
For InstructBLIP, the options of '-d' are ['lang','encoder','qformer','query'].

## Generation

Generation response to VQA datasets by running command:

```shell
python generate.py -m llava
```

## Evaluation

The evaluation code can be found in evaluate.py. The input file should contain three elements: ground_truth, answer and index. Evaluate model performance by running the command:

```shell
python evaluate.py -m llava
```

## Logit Lens

Run the command

```shell
python logit_len.py -m llava
```

to investigate the hidden states of intermedia layers.

## Cite

```bibtex
@article{huo2024mmneuron,
  title={MMNeuron: Discovering Neuron-Level Domain-Specific Interpretation in Multimodal Large Language Model},
  author={Huo, Jiahao and Yan, Yibo and Hu, Boren and Yue, Yutao and Hu, Xuming},
  journal={arXiv preprint arXiv:2406.11193},
  year={2024}
}
```
