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