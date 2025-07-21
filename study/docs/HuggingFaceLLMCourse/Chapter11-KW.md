# 11. Supervised Fine-Tuning
- 과거에는 summerization과 QA와 같은 specific task를 finetune하였으나, **최근엔 broad range of task를 동시에 finetune하는 것이 더 일반적이다.** 이것이 STF로 잘 알려진 방식이다.
- 이 Process를 통해 모델이 더 다양하고 많은 case를 다룰 수 있도록 돕는다.
    - 대부분의 LLM들은 Human Preferences에 맞춰 align되고 helpful하도록 SFT되어있음

### 이번 장의 구성
1. Chat Template
    - user와 AI 모델 사이에서 interaction하는 Chat template 구조, 일관성, 문맥적으로 적절한 응답 보장
    - Chat template은 system prompt와 role-based message와 같은 component를 포함함
2. Supervised Fine-Tuning
    - SFT는 pre-trained language model을 specific task에 adapting하는 critical process
    - SFT는 레이블이 지정된 예제가 있는 task-specific dataset에 대한 모델 학습이 포함된다.
3. Low Rank Adaption
    - low-rank matrices를 model의 layer에 추가함으로써 language model을 finetuning하기 위한 technique
    - LoRA는 model's pretrained knowledge를 보존하여 efficient fine tuning을 할 수 있게 함.
    - LoRA의 주요 이점 중 하나 : significant memory saving
4. Evaluation
    - task-specific dataset에서 model의 performance 측정

## Chat Templates
### Introduction
- Chat templates는 language model과 user 사이에서 구조화된 interaction을 위해 필수적
- 본 장에선 chat template이 무엇인지, 그게 중요한 이유, 효과적으로 사용하는 방법에 대해 탐구할 예정

### Model Types and Templates
- Base Models vs Instruct Models
    - base model은 next token을 예측하기 위한 raw text data에서 훈련된 모델을 말함.
    - 반면에 instruct model은 구체적으로 instruction을 따르고 대화에 관여하기 위해 finetune된 모델이다.
- Instruction tuned model들은 specific conversational structure를 따르도록 훈련되어 있음. 그래서 complex interaction을 다룰 수 있고 tool use, multimodal input, function calling을 포함한다.
- base model을 instruct model처럼 행동하게 하기 위해서, 우리는 Model이 이해할 수 있는 일관된 방식으로 프롬프트를 format해야 한다. 

### Common Template Formats
- specific implementations을 나누기 전에, 모델들이 기대하는 대화형식을 이해하는 것은 중요함.
- 우리가 아래같이 conversation을 구조화한다면
    ```python
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi! How can I help you today?"},
        {"role": "user", "content": "What's the weather?"},
    ]
    ```
    - ChatML template에서는 아래처럼 인식된다. (SmolLM2, Qwen2)
        ```python
        <|im_start|>system
        You are a helpful assistant.<|im_end|>
        <|im_start|>user
        Hello!<|im_end|>
        <|im_start|>assistant
        Hi! How can I help you today?<|im_end|>
        <|im_start|>user
        What's the weather?<|im_start|>assistant
        ```
    - mistral template에서는 아래처럼 인식된다.
        ```python
        <s>[INST] You are a helpful assistant. [/INST]
        Hi! How can I help you today?</s>
        [INST] Hello! [/INST]
        ```
    - 이러한 format 사이의 key difference는 아래의 사항을 포함한다.
        1. System Message Handling
            - Llama2는 `<<SYS>>`로 감싸서 표현한다.
            - Llama3는 `<|system|>`로 시작하고 `</s>`로 끝낸다.
            - Mistral은 위에서 보이는 것과 같이 첫 instruction으로 system prompt를 제공한다.
            - Qwen도 위에서 보이는 것 같이 `<|im_start|>`뒤에 system이라고 명시적으로 표현한다.
            - ChatGPT는 `SYSTEM:` prefix를 이용한다.
        2. Message Boundaries
            - Llama2는 `[INST]`와 `[\INST]`태그를 이용한다.
            - Llama3는 role을 구체적으로 명시한 태그(`<|system|>, <|user|>, <|assistant|>`)와 종료 태그 `</s>`를 함께 이용한다.
            - Mistral은 위에서 보이는 것과 같이 `[INST]`와 `[\INST]`, `<s>`와 `</s>`를 함께 사용한다.
            - Qwen도 위에서 보이는 것처럼 role을 구체적으로 명시한다.
        3. Special Tokens
            - Llama2는 `<s>`와`</s>`로 전체 대화 boundaries를 감싼다.
            - Llama3는 `</s>`를 사용해서 각 메시지의 end point를 나타낸다.
            - Mistral은 각 turn에 대해 `<s>`와`</s>` 사용
            - Qwen은 role-specific start,end token 사용
        - Chat Template을 보는 방법
            - AutoTokenizer를 이용해서 확인 가능
            ```python
            from transformers import AutoTokenizer

            mistral_tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")

            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"},
            ]

            mistral_chat = mistral_tokenizer.apply_chat_template(messages, tokenize=False)
            ```
- Chat Template은 위와같은 기본적 대화 이외에도 아래같은 다양한 시나리오 표현 가능
    1. **Tool Use** : model이 외부 tool / API랑 interact가 필요할 때
    2. **Multimodal Inputs** : image, audio, etc. 데이터를 다룰 때
    3. **Function Calling** : Structured Function 실행
    4. **Multi turn context** : 대화 기록을 유지하기 위해

### Best Practice
- 일반적 Guidelines
    1. **Consistent Formatting** : 항상 같은 Conversation Fortmat을 사용하기
    2. **Clear Role Definition** : 역할을 명확하게 정의하기
    3. **Context Management** : 대화 기록 유지 시, token limit에 유의하기
    4. **Error Handling** : tool call, multimodal input에서 적절한 error handling 포함하기
    5. **Validation** : Model에 input을 넣기 전, Message Structure 확인

- 예시 : smoltalk dataset을 ChatML format으로 가져오기
    1. 데이터 로드
        ```python
        from datasets import load_dataset

        dataset = load_dataset("HuggingFaceTB/smoltalk")
        ```
    2. processing function 제작
        ```python
        def convert_to_chatml(example):
            return {
                "messages": [
                    {"role": "user", "content": example["input"]},
                    {"role": "assistant", "content": example["output"]},
                ]
            }
        ```
    3. 내가 선택한 model의 tokenizer에 ChatTemplate 적용하기

## Supervised Fine-Tuning (SFT)
- SFT는 pre-trained language model이 instruction을 따르도록 하는데 사용되는 process
- pre-trained language model 자체도 general capabilities를 갖지만, SFT는 거기에 더해 더 잘 대화를 이해하고 prompt를 이해할 수 있도록 돕는다.
- 해당 단락에서는 Deepseek-R1-Distill-Qwen-1.5B 모델을 SFTTrainer를 사용하여 SFT하는 것에 대해 가이드할 예정

### When to use SFT?
- SFT를 시작하기 전, 첫 단계로 현존하는 Instruction Model이 내 use case에 충분한지 아닌지 고려해야만 한다.
    - Computational Resource가 많이 소요되고, Engineering Effort가 많이 요구되기 때문

- SFT가 필수적이라고 결정했으면 진행 여부는 2가지 요인에 의존한다.
    1. Template Control
        - SFT는 모델의 output 구조를 넘어 precise control을 가능하게 함. 특히 이는 아래 상황에 유용하다.
            1. specific chat template format 생성
            2. strict output schemes를 따르도록 함
            3. response 전반에 걸쳐 일관된 style 유지
    2. Domain Adaption
        - 특정 분야에서 동작해야 할 때, SFT는 model이 domain에 align되도록 도움.
            1. 도메인 용어 및 개념 교육
            2. professional standard 강화
            3. 적절하게 technical queries 다루기
            4. industry-specific guideline 따르기
    - 종합적으로 SFT를 적용하기 전, 아래의 사항을 요구하는지 아닌지 평가하기
        1. Precise Output Format
        2. Domain Specific Knowledge
        3. Consistent Response Pattern
        4. Adherence to Specific Guidelines
### Dataset Preparation
- SFT는 input-output pair로 구성된 task specific dataset을 요구한다.
    - 각 pair는 아래의 요소로 구성된다.
        1. An input prompt
        2. The expected model response
        3. Any additional context or metadata
    - Training data의 quality는 성공적인 Finetuning을 위해 매우 중요

### Training Configuration
- Finetuing의 성공을 위해서 올바른 training parameter를 선택하는 것은 매우 중요함.
- SFTTrainer's parameter
    1. Training Duration Parameters
        - `num_train_epochs` : total train epoch 조정
        - `max_steps` : epoch의 대체제, training step의 최대 횟수 설정
        - 높은 epoch은 더 나은 학습을 유도하지만 overfitting의 risk가 있다.
    2. Batch Size Parameters
        - `per_device_train_batch_size` : memory usage랑 training stability 결정
        - `gradient_accumulation_steps` : 더 큰 effective batch size를 가능하게 함
        - 큰 batchsize는 안정적인 gradient를 제공하지만 memory를 많이 요구함.
    3. Learning Rate Parameters
        - `learning_rate` : weight update의 size 결정
        - `warmup_ratio` : learning rate warmup에 사용될 training의 비율
        - Too high -> instability , Too low -> slow learning
    4. Monitoring Parameters
        - `logging_steps` : Metric logging의 빈도수
        - `eval_steps` : 얼마나 자주 validation data로 평가할지
        - `svae_steps` : model checkpoint 저장의 빈도수
    - TIP : 
        - Conservative Value로 시작하고, Monitoring을 통해 적절하게 조정
            1. 1-3 epoch으로 시작
            2. 초기엔 작은 batch size 사용
            3. 면밀하게 validation metric 모니터링
            4. Training이 안정적이지 않다면 learning rate 조정하기
### Implementation with TRL
- Transformers Reinforcement Learning (TRL)의 Library인 SFTTrainer Class를 사용하여 SFT하는 예시
```python
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
import torch

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load dataset
dataset = load_dataset("HuggingFaceTB/smoltalk", "all")

# Configure model and tokenizer
model_name = "HuggingFaceTB/SmolLM2-135M"
model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_name).to(
    device
)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)
# Setup chat template
model, tokenizer = setup_chat_format(model=model, tokenizer=tokenizer)

# Configure trainer
training_args = SFTConfig(
    output_dir="./sft_output", # SFT 결과 저장 
    max_steps=1000, # 최대 Training step
    per_device_train_batch_size=4, # Device 당 train_batch_size 
    learning_rate=5e-5, # lr
    logging_steps=10, # 10회마다 logging
    save_steps=100, # 100회마다 저장
    eval_strategy="steps", # 모델을 언제 평가할지 기준 (step-> step이 끝날 때 기준, epochs -> epoch이 끝날 때 기준, no)
    eval_steps=50, # 50회 마다 평가
)

# Initialize trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    processing_class=tokenizer,
)

# Start training
trainer.train()
```

### Packing the Dataset
- SFTTraining의 example packing이라는 기능을 제공
    - example packing이란, 여러 개의 짧은 example을 동일한 input sequence에 packing하여 학습 중 GPU Untilization 최대화에 기여함.
        - 간단하게 말해서 Sample의 길이가 10,16,20 이런식인데 max_length가 512라서 padding이 많이 드는경우, 이러한 짧은 sample들을 합쳐서 넣는다는 것
    - SFTTrainer의 parameter 중 packing 값을 True로 설정해서 사용 가능.

- Packed Dataset을 `max_steps`과 사용할 때, 예상한 packing configuraion보다 더 많은 epoch을 사용해야 할 수도 있음
- Packing은 QA같은 multiple field를 가진 dataset으로 동작할 때 유용함.
- Evaluation Dataset에 대해서는 SFT Config에서 eval_packing을 False로 설정함으로써, packing을 비활성화할 수 있음


- 예시 
    ```python
    # Packing을 True로 설정
    training_args = SFTConfig(packing=True)

    trainer = SFTTrainer(model=model, train_dataset=dataset, args=training_args)

    trainer.train()
    ```
    - packing dataset을 쓸 때는 field들을 결합해 하나의 input sequence로 만드는 custom formatting function을 정의해야 한다.
        - function은 example list를 받아서 return한다.
    ```python
    def formatting_func(example):
        text = f"### Question: {example['question']}\n ### Answer: {example['answer']}"
        return text


    training_args = SFTConfig(packing=True)
    trainer = SFTTrainer(
        "facebook/opt-350m",
        train_dataset=dataset,
        args=training_args,
        formatting_func=formatting_func,
    )
    ```
### Monitoring Training Progress
- Loss Pattern 이해
    - Training Loss는 일반적으로 3가지 phase를 따른다.
        1. Initial Sharp Drop : 빠르게 새로운 데이터 분포에 대해 adaption
        2. Gradual Stabilization : Model이 Finetune됨에 따라 학습 속도 느려짐
        3. Convergence : Loss가 매우 안정적이게 되고, training completion을 나타냄
        ![Training](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/nlp_course_sft_loss_graphic.png)
- Metrics to Monitor
    - 효과적인 모니터링은 quantitive metric을 추적하고, qualitive metric을 평가하는 것을 포함함.
    - 이용 가능한 Metric
        1. Training Loss
        2. Validation Loss
        3. Learning Rate Progression
        4. Gradient Norms
        - Training 중 3가지 주의해야할 요소 
            1. Training Loss는 감소하는데 Validation Loss는 증가함 (overfitting)
            2. Validation Loss가 Significantly 감소하지 않음 (underfitting)
            3. Extremely low loss values (potential memorization) 
            4. Inconsistent output formatting (template learning issue)

### The Path to Convergence
- Training 과정동안 Loss Curve는 점점 안정적으로 변한다.
    - healthy training의 **key indicator는 training loss와 validation loss의 small gap**
        - 일반화가 잘 돼고 있다는 증거, 이때 gap의 absolute value는 dataset,task에 따라 다름.

- Monitoring Training Progress - 각 상황별 전략
    1. Validation Loss가 Training Loss 대비 상당히 천천히 감소하는 경우 => Overfitting
        ![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/sft_loss_1.png)
        - Reducing the training steps
        - Dataset size 키우기
        - Validation data quality를 높이고, 다양하게 하기
    2. Loss 자체가 상당한 향상이 없는 경우
        ![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/sft_loss_2.png)        
        - Learning too slowly (learning rate 증가하기)
        - Struggling with the task (data quality, task 복잡성 확인하기)
        - Hitting Architecture Limitations (consider a different model)
    3. 극단적으로 낮은 loss값
        ![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/sft_loss_3.png)
        - 모델이 새로운, 유사한 것에 대해 안좋은 성능
        - output의 다양성 부족
        - training example이랑 너무 유사

### Evaluation after SFT
- SFT가 완료되면 다음의 action을 고려하기
    1. test model로 평가
    2. 다양한 input에 걸쳐 template을 잘 준수하는지 확인
    3. Test Domain Specific knowledge retention
    4. Real-World Performance Metric 모니터링

## LoRA
- Attention weight앞에 matrix들을 붙이고 최적화하여 효율적으로 학습하는 방식 (SFT대비 90% 이상 자원 절약)
- PEFT중에 한 방식으로 pre-trained model의 weight을 freeze한 뒤, model의 layer에 훈련가능한 rank decomposition matrix들을 추가해서 학습
    - 그래서 전체 parameter를 모두 tuning하는 대신, low-rank decomposition을 통해 가중치 업데이트를 더 작은 matrices로 분해야하여 진행 (학습 가능한 파라미터 수를 매우 줄임)
### LoRA의 Key Advantage
1. Memory Efficiency
    - 오직 Adapter Parameter만 GPU Memory에 저장된다.
    - Base Model weight은 frozen되고 낮은 precision으로 load해도 된다.(Memory Usage 감소)
    - Consumer GPU(일반 GPU)에서도 큰 모델을 Finetuning할 수 있음
2. Training Features
    - minimal setup으로 Native PEPT/LoRA Intergration 가능
    - QLoRA로 더욱 Memory 효율적으로 사용 가능
3. Adapter Management
    - Checkpoint 동안 Adapter weight 절약
    - Adapter를 다시 Base Model에 합칠 수 있음
### Loading LoRA Adapters with PEFT
- PEFT라는 것 자체도 PEPT Method들을 관리하고 loading하는 unified interface를 공급하는 라이브러리
- 다른 PEFT Method들 사이에서, 쉽게 load하고 switch할 수 있게 함.

- PEFT 라이브러리 사용법
    - `load_adapter`를 사용하여 adapter를 쉽게 load할 수 있다. 
    - `set_adapter`를 사용하여 adapter를 설정할 수 있다.
    - `unload`를 사용하여 LoRA 모듈을 unload한 base model을 return 받을 수 있다.
    ```python
    from peft import PeftModel, PeftConfig

    config = PeftConfig.from_pretrained("ybelkada/opt-350m-lora")
    model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
    lora_model = PeftModel.from_pretrained(model, "ybelkada/opt-350m-lora")
    ```

### Fine-tune LLM using trl and the SFTTrainer with LoRA
- trl의 SFTTrainer는 LoRA Adapter와의 통합을 제공함. 따라서 우리는 LoRA를 SFT에서 한 것처럼 Finetune 가능
- 본 장에서는 PEFT 라이브러리의 `LoRAConfig`Class를 사용할 예정. 다음은 setup에서 요구하는 몇가지 configuration setting하는 과정
    1. Define LoRA Configuration(rank, alpha, dropout)
    2. PEFT Config를 바탕으로 SFTTrainer 생성
    3. Adapter weight을 훈련하고 저장

### LoRA Configuration
- key parameters
- r (rank)
    - weight update에 사용할 low-rank matrices의 차원
    - 일반적으로 4-32를 사용하고, 값이 낮을 수록 Compression은 뛰어나나, Expressiveness는 떨어진다.
    - EX) $W_0$= 4096x4096, r = 8이라면
        - $A$ = 8x4096
        - $B$ = 4096x8
- lora_alpha
    - LoRA layer의 Scaling factor, 일반적으로 r값의 2배 사용
    - 값이 높을수록 adaption 효과가 높아짐
    - $output = W*x + (LoRA_{alpha}/r) * B*A*x$ 에 따라 lora_alpha값이 커지면, output의 변화가 큼
- lora_dropout
    - LoRA Layer의 dropout 확률, 일반적으로 0.05-0.1 사용
    - 값이 높을수록 overfitting을 막는데 도움을 줌
- bias	
    - bias term의 훈련 control
    - 3가지 옵션 `none`, `all`, or `lora_only`
    - `none`이 memory 효율을 위해 보통 none 선택
- target_modules
    - LoRA를 적용할 model module 정함
    - `all-linear` or `q_proj`,`v_proj`등이 있다.
    - 많은 모듈을 설정할 수록 메모리 사용량 증가

### Using TRL with PEFT
- PEFT method는 TRL과 합쳐질 수 있음. 여기서 LoRA Config를 모델을 로딩할 때 전달할 수 있다.
    ```python
    from peft import LoraConfig

    rank_dimension = 6
    lora_alpha = 8
    lora_dropout = 0.05

    peft_config = LoraConfig(
        r=rank_dimension,  # Rank dimension - typically between 4-32
        lora_alpha=lora_alpha,  # LoRA scaling factor - typically 2x rank
        lora_dropout=lora_dropout,  # Dropout probability for LoRA layers
        bias="none",  # Bias type for LoRA. the corresponding biases will be updated during training.
        target_modules="all-linear",  # Which modules to apply LoRA to
        task_type="CAUSAL_LM",  # Task type for model architecture
    )
    ```
- 다음으로 SFTTrainer에 위에서 정의한 LoRA Config를 전달함
    ```python
    # Create SFTTrainer with LoRA configuration
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        peft_config=peft_config,  # LoRA configuration
        max_seq_length=max_seq_length,  # Maximum sequence length
        processing_class=tokenizer,
    )
    ```
### Merging LoRA Adapters
- 쉬운 배포를 위해 LoRA 훈련이 끝나면 BaseModel로 통합해야한다.
- LoRA와 Model을 둘 다 load해야 하므로 CPU/GPU 메모리가 모두 충분해야하고 device auto를 선택하여 자동으로 적절한 하드웨어를 선택하도록 한다.
- Merge LoRA Adapter to Base Model
    ```python
    import torch
    from transformers import AutoModelForCausalLM
    from peft import PeftModel

    # 1. Load the base model
    base_model = AutoModelForCausalLM.from_pretrained(
        "base_model_name", torch_dtype=torch.float16, device_map="auto"
    )

    # 2. Load the PEFT model with adapter
    peft_model = PeftModel.from_pretrained(
        base_model, "path/to/adapter", torch_dtype=torch.float16
    )

    # 3. Merge adapter weights with base model
    merged_model = peft_model.merge_and_unload()
    ```
- 저장된 모델에서 크기 불일치(size discrepancies)가 발생한다면, tokenizer를 같이 저장해보기
    ```python
    # Save both model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("base_model_name")
    merged_model.save_pretrained("path/to/save/merged_model")
    tokenizer.save_pretrained("path/to/save/merged_model")
    ```

## Evaluation
- SFT, LoRA 등 훈련이 끝났으면, 평가하는 과정이 필수적이다.
    - 이 평가를 위해 engineer는 항상 도메인과 관련된 평가 데이터셋을 유지해야한다.
- 해당 단락에서는 일반적으로 평가하는 방식과 custom benchmark를 생성하는 방법을 설명한다.

### Automatic Benchmarks
- 다양한 작업 등에 적용가능한 표준화된 벤치마크, 하지만 특정 도메인의 능력 등은 측정하기 어려움

### Understanding Automatic Benchmarks
- 미리 정의된 작업과 평가 지표가 포함된 큐레이션된 데이터 세트로 구성
- 기본적 응답부터 복잡한 추론까지 평가 가능
- Automatic Benchmark의 이점 : 다양한 모델간의 일관된 비교가 가능하고, 재현 가능한 결과 제공
    - 하지만, 벤치마크 성능의 향상이 무조건 실제 세계에서 잘 동작하는 것과 연관되지는 않음을 알아야함.
### General Knowledge Benchmarks
- MMLU : 과학에서 인문학까지 57개의 주제의 지식 테스트
    - 포괄적이기 때문에, 구체적인 도메인의 깊이있는 지식을 반영하지는 못한다.
- TruthfulQA : 일반적인 잘못된 정보를 생산하는 모델의 경향을 테스트
    - 모든 잘못된 정보를 테스트하는건 X(현실적으로 당연함)

### Reasoning Benchmarks
- BBH, GSM8K : 복잡한 reasoning task 테스트
    - BBH : 논리적 생각과 계획 평가
    - GSM8K : 수학적 문제 해결에 집중
- 위 벤치마크는 분석 능력을 평가하는데에는 도움이 되지만, 실제 시나리오에서 요구되는 미묘한 추론을 Capture하기는 어려움

### Language Understanding
- HELM : 전체적인 평가 프레임워크
    - 상식, 세계지식, 추론 등의 측면에서의 능력을 평가
    - 하지만 자연스러운 대화나 도메인 specific 용어는 완전히 나타내지못함

### Domain-Specific Benchmarks
- Math Benchmark  : 수학쪽
- HumanEval Benchmark : 프로그래밍 관련
- Alpaca Eval : instruction-following 모델 능력 평가

### Alternative Evaluation Approaches

1. LLM as Judge
    - LLM을 사용하여 평가
    - 기존 보다 상세한 피드백이 가능하지만, 모델 자체의 편향과 한계가 존재
2. Evaluation Arenas
    - Crowedsourced Feedback을 통해 LLM을 평가하는 방식(ex, Chatbot Arena -> voting)
    - 이렇게하면 real-world usage를 더 잘 capture할 수 있음 (실제로 좋은거)
3. Custom Benchmark Suites
    - 조직에서 직접 개발한 내부 벤치마크 스위트, 실제 배포 조건을 반영하는 도메인별 지식 테스트와 평가 시나리오가 포함된다.

### Custom Evaluation
- Standard benchmark가 baseline을 제시하지만, 이것이 유일한 평가 방식이 되어서는 안된다. Custom Evaluation을 필수!
    1. Standard benchmark로 시작하기
    2. 사용 사례의 구체적인 요구 사항과 조건을 파악하기, 모델이 실제로 어떤 작업을 수행하게 되며, 어떤 종류의 에러가 가장 문제가 될 수 있는지 파악하기
    3. 실제 사용 사례를 반영하는 맞춤형 데이터 세트를 개발
        1. 나의 도메인의 Real User Query 
        2. 내가 실제 마주한 edge case 찾기
        3. 특히 어려운 시나리오 찾기
    4. Multi-layered evaluation strategy 구현 고려
        1. 빠른 피드백을 위한 Automated Metric
        2. 인간 평가 (미묘한 이해)
        3. specialized application을 위한 Domain Expert Review
        4. A/B Test

### Implementing Custom Evaluations
- 이 섹션에서는 미세 조정된 모델에 대한 평가 구현을 다룸 
- lighteval을 사용하면 라이브러리에 내장된 다양한 작업을 포함하는 표준 평가 벤치마크에서 평가 가능
- LightEval task들은 아래와 같은 구체적 포맷을 사용하여 정의된다:
    - `{suite}|{task}|{num_few_shot}|{auto_reduce}`
        - suite : benchmark suite (mmlu, truthfulqa 등)
        - task : suite 안의 구체적인 task
        - num_few_shot : Prompt 안에 포함된 example의 수 (0 -> zeroshot)
        - auto reduce : Prompt가 너무 길면 자동적으로 줄일지 말지 (0 or 1)
    - ex, `mmlu|abstract_algebra|0|0`
        - mmlu의 abstract algebra task를 zero shot inference 평가

### Example Evaluation Pipeline
- lighteval과 vllm backend를 사용하여 평가하는 예시
```bash
lighteval accelerate \
    "pretrained=your-model-name" \
    "mmlu|anatomy|0|0" \
    "mmlu|high_school_biology|0|0" \
    "mmlu|high_school_chemistry|0|0" \
    "mmlu|professional_medicine|0|0" \
    --max_samples 40 \
    --batch_size 1 \
    --output_path "./results" \
    --save_generations true
```
- result
```bash
|                  Task                  |Version|Metric|Value |   |Stderr|
|----------------------------------------|------:|------|-----:|---|-----:|
|all                                     |       |acc   |0.3333|±  |0.1169|
|leaderboard:mmlu:_average:5             |       |acc   |0.3400|±  |0.1121|
|leaderboard:mmlu:anatomy:5              |      0|acc   |0.4500|±  |0.1141|
|leaderboard:mmlu:high_school_biology:5  |      0|acc   |0.1500|±  |0.0819|
```


