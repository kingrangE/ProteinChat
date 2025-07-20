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
root@bc7485bc1f5c:~# python main.py --basemodel_metric --lora_metric
Device = cuda
config.json: 100%|█| 850/850 [00:00<00:00
model.safetensors: 100%|█| 2.56G/2.56G [0
generation_config.json: 100%|█████████████████████████████████████████| 139/139 [00:00<00:00, 127kB/s]
tokenizer_config.json: 70.3kB [00:00, 56.8MB/s]
vocab.json: 1.93MB [00:00, 30.4MB/s]
merges.txt: 1.22MB [00:00, 37.6MB/s]
tokenizer.json: 7.91MB [00:00, 92.5MB/s]
special_tokens_map.json: 6.70kB [00:00, 16.4MB/s]
chat_template.jinja: 5.49kB [00:00, 10.8MB/s]
README.md: 53.2kB [00:00, 135MB/s]
dataset_infos.json: 138kB [00:00, 93.2MB/s]
test-00000-of-00001.parquet: 100%|███████████████████████████████| 1.04M/1.04M [00:00<00:00, 28.2MB/s]
validation-00000-of-00001.parquet: 100%|████████████████████████████| 116k/116k [00:00<00:00, 195MB/s]
dev-00000-of-00001.parquet: 100%|████████████████████████████████| 15.1k/15.1k [00:00<00:00, 48.2MB/s]
Generating test split: 100%|████████████████████████████| 1534/1534 [00:00<00:00, 98700.10 examples/s]
Generating validation split: 100%|████████████████████████| 170/170 [00:00<00:00, 27766.03 examples/s]
Generating dev split: 100%|█████████████████████████████████████| 5/5 [00:00<00:00, 933.52 examples/s]
Map: 100%|█████████████████████████████████████████████████| 767/767 [00:00<00:00, 2807.32 examples/s]
Split is done : Train 767, Test 767

===== Base Model 성능 측정 =====
Evaluating: 100%|███████████████████████████████████████████████████| 767/767 [01:23<00:00,  9.21it/s]
  - Base Model Accuracy: 0.1890

===== LoRA Fine-tuning 진행 =====
trainable params: 3,194,880 || all params: 1,282,586,368 || trainable%: 0.2491
/root/main.py:128: FutureWarning: `tokenizer` is deprecated and will be removed in version 5.0.0 for `Trainer.__init__`. Use `processing_class` instead.
  trainer = Trainer(
No label_names provided for model class `PeftModelForCausalLM`. Since `PeftModel` hides base models input arguments, if label_names is not given, label_names can't be set automatically within `Trainer`. Note that empty label_names list will be used instead.
{'loss': 4.0121, 'grad_norm': 0.6901770234107971, 'learning_rate': 2.5e-05, 'epoch': 0.21}            
{'loss': 3.8842, 'grad_norm': 1.0604897737503052, 'learning_rate': 4.342105263157895e-05, 'epoch': 0.42}
{'loss': 3.7254, 'grad_norm': 1.0036414861679077, 'learning_rate': 3.157894736842105e-05, 'epoch': 0.62}
{'loss': 3.547, 'grad_norm': 1.0301498174667358, 'learning_rate': 1.8421052631578947e-05, 'epoch': 0.83}
{'train_runtime': 93.5183, 'train_samples_per_second': 8.202, 'train_steps_per_second': 0.513, 'train_loss': 3.724924008051554, 'epoch': 1.0}
100%|█████████████████████████████████████████████████████████████████| 48/48 [01:33<00:00,  1.95s/it]
Finetuning is done

===== Fine-tuned Model 성능 측정 =====
Evaluating: 100%|███████████████████████████████████████████████████| 767/767 [01:55<00:00,  6.64it/s]
  - LoRA Model Accuracy: 0.2086

===== 최종 성능 비교 =====
Base Model Accuracy: 0.1890
LoRA Tuned Model Accuracy: 0.2086
성능 향상: +0.0196
```