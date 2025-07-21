## installation
``` bash
pip install git+https://github.com/lgai-exaone/transformers@add-exaone4 accelerate torch torchvision torchaudio peft trl tqdm pandas 
```

## Usage

### Base Model Test
``` bash
python study/code/LoRA_code/baseline/main.py --test
```

### Base Model Info - visualize model layer
``` bash
python study/code/LoRA_code/baseline/main.py --info
```

### Base Model Evaluation 
``` bash
python study/code/LoRA_code/baseline/main.py --basemodel_metric
```

### LoRA Fine Tuning & Evaluation
``` bash
python study/code/LoRA_code/baseline/main.py --lora_metric
```

### Base model compare with LoRA finetuned model
``` bash
python study/code/LoRA_code/baseline/main.py --basemodel_metric --lora_metric
```

### Baseline Execution Result
``` bash
README.md: 
 53.2k/? [00:00<00:00, 7.28MB/s]
dataset_infos.json: 
 138k/? [00:00<00:00, 9.89MB/s]
test-00000-of-00001.parquet: 100%
 1.04M/1.04M [00:00<00:00, 37.6MB/s]
validation-00000-of-00001.parquet: 100%
 116k/116k [00:00<00:00, 21.1MB/s]
dev-00000-of-00001.parquet: 100%
 15.1k/15.1k [00:00<00:00, 2.65MB/s]
Generating test split: 100%
 1534/1534 [00:00<00:00, 100510.24 examples/s]
Generating validation split: 100%
 170/170 [00:00<00:00, 25772.85 examples/s]
Generating dev split: 100%
 5/5 [00:00<00:00, 867.67 examples/s]
Map: 100%
 767/767 [00:00<00:00, 3382.00 examples/s]
Split is done : Train 767, Test 767
Device = cuda

===== Base Model 성능 측정 =====
Evaluating: 100%|██████████| 767/767 [01:23<00:00,  9.16it/s]
  - Base Model Accuracy: 0.1890

===== Prompt Tuning 진행 =====
/tmp/ipykernel_448/967472046.py:129: FutureWarning: `tokenizer` is deprecated and will be removed in version 5.0.0 for `Trainer.__init__`. Use `processing_class` instead.
  trainer = Trainer(
No label_names provided for model class `PeftModelForCausalLM`. Since `PeftModel` hides base models input arguments, if label_names is not given, label_names can't be set automatically within `Trainer`. Note that empty label_names list will be used instead.
trainable params: 40,960 || all params: 1,279,432,448 || trainable%: 0.0032
 [48/48 01:22, Epoch 1/1]
Step	Training Loss
10	3.735900
20	3.455100
30	3.307900
40	3.176500
Prompt Tuning is done

===== Fine-tuned Model 성능 측정 =====
Evaluating:   0%|          | 0/767 [00:00<?, ?it/s]/usr/local/lib/python3.10/dist-packages/peft/peft_model.py:2060: UserWarning: Position ids are not supported for parameter efficient tuning. Ignoring position ids.
  warnings.warn("Position ids are not supported for parameter efficient tuning. Ignoring position ids.")
Evaluating: 100%|██████████| 767/767 [01:35<00:00,  8.06it/s]
  - Prompt Tuned Model Accuracy: 0.2451

===== 최종 성능 비교 =====
Base Model Accuracy: 0.1890
Prompt Tuned Model Accuracy: 0.2451
성능 향상: +0.0561
```