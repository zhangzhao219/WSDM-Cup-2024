# WSDM Cup 2024

## Environment

1. Follow [Installation for modelscope/swift](https://github.com/modelscope/swift?tab=readme-ov-file#%EF%B8%8F-installation) to install swift.

2. Install [vllm](https://docs.vllm.ai/en/latest/getting_started/installation.html)

3. Install [deepspeed](https://github.com/microsoft/DeepSpeed?tab=readme-ov-file#installation)

4. Install [sklearn](https://scikit-learn.org/stable/install.html)

5. Install [SentenceTransformers](https://www.sbert.net/#installation)

Or you can run this: (Tested on V100 32G with CUDA 11.8, Ubuntu 20.04.1)

```bash
conda create -n swift python=3.10
conda activate swift
pip install ms-swift[all] -U
pip install vllm==0.3.1
pip install deepspeed
pip install scikit-learn
pip install sentence_transformers
```

Main package version:

```bash
python==3.10.13
ms-swift==1.6.1
scikit-learn==1.4.1.post1
sentence-transformers==2.3.1
torch==2.1.2
transformers==4.37.2
vllm==0.3.1
```


## Reproduce results on the leaderboard

**You can find all intermediate files in ```result``` folder**

### Prepare models

1. Download Pretrained Models From Huggingface
    - [upstage/SOLAR-10.7B-Instruct-v1.0](https://huggingface.co/upstage/SOLAR-10.7B-Instruct-v1.0) **(10.7 B)**
    - [nomic-ai/nomic-embed-text-v1](https://huggingface.co/nomic-ai/nomic-embed-text-v1) **(0.14 B)**

2. Download Our 8 Finetuned LoRA Adapters From [our huggingface repository](https://huggingface.co/zhangzhao219/WSDM_Cup_2024/tree/main/checkpoints) **(0.03 B Each)**

**So our model size is 10.7B + 0.14B + 0.03B * 8 = 11.08B, much fewer than 14 billion (14B) parameters.**

3. Put them in the right folder. The folder should look as follows:

```bash
└── checkpoints
    ├── v08-20240205-114459/
    ├── v10-20240205-114325/
    ├── v13-20240202-072530/
    ├── v13-20240206-111010/
    ├── v16-20240206-224659/
    ├── v27-20240209-133614/
    ├── v33-20240210-002918/
    └── v35-20240210-120550/
└── pretrained
    └── nomic-ai/nomic-embed-text-v1/
        ├── 1_Pooling/
        ├── config.json
        ├── config_sentence_transformers.json
        ├── configuration_hf_nomic_bert.py
        ├── .gitattributes
        ├── .locks/
        ├── modeling_hf_nomic_bert.py
        ├── model.safetensors
        ├── modules.json
        ├── onnx/
        ├── pytorch_model.bin
        ├── README.md
        ├── sentence_bert_config.json
        ├── special_tokens_map.json
        ├── tokenizer_config.json
        ├── tokenizer.json
        └── vocab.txt
    └── upstage/SOLAR-10.7B-Instruct-v1.0/
        ├── config.json
        ├── generation_config.json
        ├── .gitattributes
        ├── .locks/
        ├── model-00001-of-00005.safetensors
        ├── model-00002-of-00005.safetensors
        ├── model-00003-of-00005.safetensors
        ├── model-00004-of-00005.safetensors
        ├── model-00005-of-00005.safetensors
        ├── model.safetensors.index.json
        ├── README.md
        ├── solar_logo.png
        ├── tokenizer_config.json
        ├── tokenizer.json
        └── tokenizer.model
```

### Inference Result

Run ```python data_format.py``` to preprocess original test data.

Then run shell script in the ```runsh``` folder

```bash
bash runsh/v08-20240205-114459.sh
bash runsh/v10-20240205-114325.sh
bash runsh/v13-20240202-072530.sh
bash runsh/v13-20240206-111010.sh
bash runsh/v16-20240206-224659.sh
bash runsh/v27-20240209-133614.sh
bash runsh/v33-20240210-002918.sh
bash runsh/v35-20240210-120550.sh
```

1. You can modify CUDA device at the beginning of each shell script ```CUDA_VISIBLE_DEVICES=```
2. The result files are saved in the ```merge``` folder, which should look as follows:

```bash
└── merge
    ├── v08-20240205-114459.jsonl
    ├── v10-20240205-114325.jsonl
    ├── v13-20240202-072530.jsonl
    ├── v13-20240206-111010.jsonl
    ├── v16-20240206-224659.jsonl
    ├── v27-20240209-133614.jsonl
    ├── v33-20240210-002918.jsonl
    └── v35-20240210-120550.jsonl
```

Besides, the results above are as follows:

| File | Word-level ROUGE-L | Character-level ROUGE-L | Keywords Recall |
|--------| ---------|--------|--------|
| v08-20240205-114459 | 0.45532953438881013 | 0.6143454883849857 | 0.6824189095928223 |
| v10-20240205-114325 | 0.456275615214309 | 0.6149276913541135 | 0.6817805383022769 |
| v13-20240202-072530 | 0.4554468517276402 | 0.6141346993379754 | 0.6827095609704305 |
| v13-20240206-111010 | 0.456388581088847 | 0.6149210447203279 | 0.6840088655306036 |
| v16-20240206-224659 | 0.45375515045837794 | 0.613359666771279 | 0.6879538939321544 |
| v27-20240209-133614 | 0.45574561117381773 | 0.6145520850027292 | 0.6826942984551678 |
| v33-20240210-002918 | 0.4559195951083145 | 0.6141543510329665 | 0.6865596963423041 |
| v35-20240210-120550 | 0.45573339341665703 | 0.614208192382808 | 0.6813332802463232 |


So even if they are not ensembled, each of them is still way ahead of the second place.

### Ensemble

First, calculate the embedding score

```bash
python calculate_score.py
```

Note that this program is accelerated by ```torch.multiprocessing```, you can modify the number of processes near ```num_group = 16```. (It works well in V100 32G)

Then generate final result,

```bash
python merge_score.py
```

It will generate ```emb_a_s_8_0_1_2_3_4_5_6_7.zip``` in the root folder, which is our final result.

| Word-level ROUGE-L | Character-level ROUGE-L | Keywords Recall |
|---------|--------|--------|
| 0.465360141853671 | 0.6208371209722543 | 0.6953475871954128 |

## Contacts

Zhao Zhang: [zhangzhao22s@ict.ac.cn](mailto:zhangzhao22s@ict.ac.cn)
