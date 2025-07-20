# Prompt tuning
![Prompt Tuning](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/peft/prompt-tuning.png)
- 매우 적은 양의 task-specific prompt parameter를 훈련하고 저장함
- Prompt Tuning의 경우 T5 model에서 classification task를 빠르게 수행하기 위해 개발되었으며, 모든 downstream task는 Text Generation으로 연결됨.
    - 예를 들어, sequence classification 작업은 보통 sequence에 single class label을 할당한다. 
    - text generation task로 casting 하면서, class label을 구성하는 토큰이 생성된다.
    - Prompt는 token의 series로 input에 추가된다. 일반적으로 model의 파라미터들은 fixed되며, 이는 prompt tokens 또한 model parameters에 fixed됨을 의미한다.

- Key Idea
    - Prompt Tuning은 Prompt token들은 독립적으로 update되는 their own parameters를 갖는다는 것
        - pretrained model의 parameter를 frozen하고 **오직 update하는건 prompt token embedding의 gradients라는 것**
    - 이렇게 학습한 결과는 전체 모델을 training하는 traditional method와 비슷하고 model의 size가 커질수록 prompt tuning performance의 규모가 커진다.

## Prompt Tuning for Causal Language Modeling - 코드
- [HuggingFace Docs Prompt Tuning](https://huggingface.co/docs/peft/main/en/task_guides/clm-prompt-tuning)

### Setup
- 사전에 정의해야 하는 것
    1. Model
    2. Tokenizer
    3. Dataset
    4. Dataset Column (훈련에 사용될)
    5. Training Parameters
    6. Prompt Tuning Config
        - 아래의 요소를 가짐
            - task type에 대한 정보
            - prompt embedding을 초기화하기 위한 text
            - virtual token의 수
            - 사용할 tokenizer
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator, get_linear_schedule_with_warmup
from peft import get_peft_config, get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType, PeftType
import torch
from datasets import load_dataset
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
# Device 설정
device = "cuda"
# model set
model_name_or_path = "bigscience/bloomz-560m"
tokenizer_name_or_path = "bigscience/bloomz-560m"
# prompt tuning config set
peft_config = PromptTuningConfig(
    #task type : GPT Style의 언어 모델
    task_type=TaskType.CAUSAL_LM, 
    #prompt embedding을 초기화 하는 방법 : text기반 초기화 방식-> 의미있는 text로 초기화하여 더 빠른 수렴, 안정적 학습 가능 (Random초기화도 가능)
    prompt_tuning_init=PromptTuningInit.TEXT, 
    #learnable virtual token의 수(상대적으로 적은 수의 파라미터로 PEFT): 이 토큰들이 모델의 입력 앞에 붙어 모델의 행동을 조정(토큰 수가 적을 수록 메모리 사용량과 계산량이 감소)
    num_virtual_tokens=8, 
    #virtual token을 초기화할 때 사용할 text (작업 목적 나타냄) (이 text는 토큰화어서 text앞에 붙음)
    prompt_tuning_init_text="Classify if the tweet is a complaint or not:", 
    tokenizer_name_or_path=model_name_or_path, #사용할 tokenizer
)

#dataset set
dataset_name = "twitter_complaints"
checkpoint_name = f"{dataset_name}_{model_name_or_path}_{peft_config.peft_type}_{peft_config.task_type}_v1.pt".replace(
    "/", "_"
)
# 훈련에 사용할 column
text_column = "Tweet text"
label_column = "text_label"
# hyper params set
max_length = 64
lr = 3e-2
num_epochs = 50
batch_size = 8
```

### Load Dataset
- 이 단락에선 RAFT 데이터셋의 일부인 twitter_complaints 데이터셋을 사용한다.
    - tweets와 label(complaint/not complaint)로 이루어짐
```python
dataset = load_dataset("ought/raft", dataset_name)
# dataset["train"][0] {"Tweet text": "@HMRCcustomers No this is my first job", "ID": 0, "Label": 2}
```
- Label Column을 읽을 수 있게 하기 위해서 일치하는 label text로 대체하고 text_label column에 저장해야함
    - map function을 사용하여 한 번에 change 할 수 있음
```python
classes = [k.replace("_", " ") for k in dataset["train"].features["Label"].names]
dataset = dataset.map(
    lambda x: {"text_label": [classes[label] for label in x["Label"]]},
    batched=True,
    num_proc=1,
)
dataset["train"][0]
{"Tweet text": "@HMRCcustomers No this is my first job", "ID": 0, "Label": 2, "text_label": "no complaint"}
```

### Preprocess dataset
- tokenizer를 setup
    - padding sequence에서 사용할 padding token을 configure하고, token화된 레이블의 최대 길이 결정
```python
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path) #model의 tokenizer load
if tokenizer.pad_token_id is None: #padding token id가 없다면 
    tokenizer.pad_token_id = tokenizer.eos_token_id #eos(종료) token을 padding token으로 설정
target_max_length = max([len(tokenizer(class_label)["input_ids"]) for class_label in classes])  #최대 길이는 input text를 토큰화한 길이의 최대로 설정
print(target_max_length)
```

- `preprocess_function` 만들기
    1. input text와 label을 tokenize
    2. batch의 각 example에 대해, padding_token_id로 padding한다.
    3. input text와 label을 model_inputs로 결합
    4. labels과 model_inputs에 대해 seperate attention mask를 생성
    5. Batch의 각 example을 반복하여, input_id,label,attention_mask를 `max_length`로 padding하고 Pytorch Tensor로 변환
```python
def preprocess_function(examples):
    batch_size = len(examples[text_column]) # 전체 예시의 개수
    inputs = [f"{text_column} : {x} Label : " for x in examples[text_column]]
    targets = [str(x) for x in examples[label_column]]
    model_inputs = tokenizer(inputs)
    labels = tokenizer(targets)
    for i in range(batch_size):
        sample_input_ids = model_inputs["input_ids"][i]
        label_input_ids = labels["input_ids"][i] + [tokenizer.pad_token_id]
        # print(i, sample_input_ids, label_input_ids)
        model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
        labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids
        model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])
    # print(model_inputs)
    for i in range(batch_size):
        sample_input_ids = model_inputs["input_ids"][i]
        label_input_ids = labels["input_ids"][i]
        model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
            max_length - len(sample_input_ids)
        ) + sample_input_ids
        model_inputs["attention_mask"][i] = [0] * (max_length - len(sample_input_ids)) + model_inputs[
            "attention_mask"
        ][i]
        labels["input_ids"][i] = [-100] * (max_length - len(sample_input_ids)) + label_input_ids
        model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:max_length])
        model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:max_length])
        labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:max_length])
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
```
-  위의 preprocess dataset을 map함수를 이용하여 전체 데이터셋에 적용할 수 있다.
```python
processed_datasets = dataset.map(
    preprocess_function,
    batched=True,
    num_proc=1,
    remove_columns=dataset["train"].column_names,
    load_from_cache_file=False,
    desc="Running tokenizer on dataset",
)
```
- train, eval dataset으로부터 DataLoader를 생성
    - `pin_memory = True`로 설정하게 되면 CPU에 있는 Dataset을 GPU로 빠르게 data transfer할 수 있다.s
```python
train_dataset = processed_datasets["train"]
eval_dataset = processed_datasets["test"]

train_dataloader = DataLoader(
    train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True
)
eval_dataloader = DataLoader(eval_dataset, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True)
```
### Train
- 이제 거의 모든 준비 끝. 
- PEFT Model을 가져오기 위해 peft_config와 model을 `get_peft_model()` 함수에 전달
```python
model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
model = get_peft_model(model, peft_config)
print(model.print_trainable_parameters())
"trainable params: 8192 || all params: 559222784 || trainable%: 0.0014648902430985358"
```
- optimizer와 learning rate scheduler를 셋업한다.
```python
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=(len(train_dataloader) * num_epochs),
)
```
- GPU로 모델을 이동하고 훈련을 시작하기 위한 training loop를 작성하기
```python
model = model.to(device) #GPU로 모델 이동

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for step, batch in enumerate(tqdm(train_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()} #batch에 있는 tensor를 GPU로 이동
        outputs = model(**batch) #batch를 model에 전달하여 output 얻기
        loss = outputs.loss #output loss get
        total_loss += loss.detach().float() # total loss에 현재 loss를 더함. 이때, detach를 이용하여 연산 기록을 추적하지 않도록 함
        loss.backward() # backpropagation을 통해 gradient 계산
        optimizer.step() # gradient를 이용하여 weight update
        lr_scheduler.step() # lr scheduler update
        optimizer.zero_grad() # optimizer 초기화

    model.eval()
    eval_loss = 0
    eval_preds = []
    for step, batch in enumerate(tqdm(eval_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        loss = outputs.loss
        eval_loss += loss.detach().float()
        eval_preds.extend(
            tokenizer.batch_decode(torch.argmax(outputs.logits, -1).detach().cpu().numpy(), skip_special_tokens=True)
        )

    eval_epoch_loss = eval_loss / len(eval_dataloader)
    eval_ppl = torch.exp(eval_epoch_loss)
    train_epoch_loss = total_loss / len(train_dataloader)
    train_ppl = torch.exp(train_epoch_loss)
    print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}")
```
### Share Model
- 이렇게 만든 모델은 저장한 후, 원한다면 huggingface hub에 업로드할 수 있다.
1. login to hub
    ```python
    from huggingface_hub import notebook_login

    notebook_login()
    ```
2. `push_to_hub` 함수를 사용하여 hub로 upload하기
    ```python
    peft_model_id = "your-name/bloomz-560m_PROMPT_TUNING_CAUSAL_LM"
    model.push_to_hub("your-name/bloomz-560m_PROMPT_TUNING_CAUSAL_LM", use_auth_token=True)
    ```
### Inference
- inference를 위해 sample input을 model에 넣기
- 방금 전에 hub에 올린 model의 repo를 보면, `adapter_config.json` 파일을 확인할 수 있는데, 이 파일을 PeftConfig로 load하여 `peft_type`과 `task_type`를 specify한다.
    - 그리고 prompt tuned model weight와 configuration을 from_pretrained()를 이용하여 load할 수 있다.
```python
from peft import PeftModel, PeftConfig

peft_model_id = "stevhliu/bloomz-560m_PROMPT_TUNING_CAUSAL_LM"

config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
model = PeftModel.from_pretrained(model, peft_model_id)
```
- tweet(dataset에 있는거)을 가져와서 token화 하기
```python
inputs = tokenizer(
    f'{text_column} : {"@nationalgridus I have no water and the bill is current and paid. Can you do something about this?"} Label : ',
    return_tensors="pt",
)
```
- 모델을 GPU에 올리고 predicted label 생성
```python
model.to(device)

with torch.no_grad():
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model.generate(
        input_ids=inputs["input_ids"], 
        attention_mask=inputs["attention_mask"], 
        max_new_tokens=10, 
        eos_token_id=3
    )
    print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True))
[
    "Tweet text : @nationalgridus I have no water and the bill is current and paid. Can you do something about this? Label : complaint"
]
```