# HuggingFace LLM Course
- [Introduction to Reinforcement Learning and its Role in LLMs](https://huggingface.co/learn/llm-course/chapter12/2)

## What is Reinforcement Learning?

### 현실 세계의 예시

- 만약 내가 강아지에게 "앉아"를 훈련시킨다면?
    1. 강아지에게 앉으라고 말함
    2. 강아지가 앉음 -> 보상
    3. 강아지가 앉지 않음 -> 알려주거나 다시 시도함
    4. 이를 반복하며 강아지는 앉아를 배우게 된다.
- Reinforcement Learning에서는 
    1. Feedback을 **reward**라고 부름
    2. 강아지 대신 **Language Model**을 훈련 (Reinforcement Learning에서 이것을 agent라고 부름)
    3. 나(피드백 주는 사람) 대신 **environment를 가짐**
    ![RL in LLM](https://huggingface.co/reasoning-course/images/resolve/main/grpo/3.jpg)
- RL을 key piece들로 나누면
    1. Agent
        - **Learner**
            - 위의 강아지 예시에서 강아지
        - LLM에선 LLM은 그자체로 우리가 훈련하기 원하는 agent가 된다.
            - Agent는 환경과 보상으로부터 학습하고 결정을 내린다.
    2. Environment
        - **Agent가 살고, 상호작용하는 세계**
            - 강아지를 예시로 할 때, 집과 내가 environment가 됨
        - LLM에서는 environment는 조금 더 추상적
            - user와 상호작용하는 것일수도 있고, 우리가 setup해놓은 simulated scenario가 될 수도 있음
        - Environment는 Agent에게 Feedback을 줌
    3. Action
        - agent가 envirionment에서 하는 선택들
            - 강아지 예시에서, action은 앉아, 서, 기다려 등
        - LLM에서는 문장을 생성하는거나, 질문에 대한 정답을 선택하는거, 대화에 어떻게 응답할지 결정하는 것 등이 있음
    4. Reward
        - environment가 action을 취한 agent에게 주는 feedback 
            - Reward는 보통 숫자
        - **Positive Rewards**
            - treats and praise
            - 잘했다는 의미
        - **Negative Rewards(penalties)**
            - no
            - 지금한건 틀렸다 다시 해봐라 하는 의미
        - LLM에서 reward는 얼마나 LLM이 specific task에서 잘 했는지를 반영하도록 design 
    5. Policy
        - Agent가 action을 선택하는 strategy
            - 내가 앉으라고 했을때 강아지가 무엇을 해야할지 이해하는 것과 같음
        - RL에선, 우리가 정말로 학습하고 개선시키려고 하는 것은 policy
            - agent에게 different situation에서 무슨 행동을 취해야하는지 agent에게 말해주는 **rule, function의 집합**
        - 초기에 policy는 random이지만 agent가 학습하면서 policy는 더 나은 action(higher reward를 얻을 수 있는)을 선택하게 된다.
        
### The RL Process: Trial and Error

![RL Porcess Example](https://huggingface.co/reasoning-course/images/resolve/main/grpo/1.jpg)
- RL은 trial and error의 과정을 통해 발생함.

| Step | Process | Description |
|---|---|---|
|Observation| Agent가 environment 탐색 | Agent가 current state와 surroundings에 대해 정보를 얻음|
|Action| Agent가 현재 정책에 맞춰 action을 실행 | 학습된 strategy를 사용하여 agent가 다음으로 해야할 것을 결정 |
|Feedback| Environment가 agent에게 reward 제공|Agent는 action이 얼마나 좋았는지/나빴는지 feedback을 받음|
|Learning|Agent는 reward에 근거하여 policy update | Agent가 strategy를 조정함(더 높은 보상을 얻거나 낮은 보상을 피하도록)
|Iteration|과정 반복| Agent가 지속적으로 의사결정 능력을 향상시키도록 반복|

### Role of RL in Large Language Models(LLMs)

- 왜 LLM에서 RL이 중요한가?
    - LLM을 잘 학습시키는 것은 까다로움.
        - LLM 학습은 internet의 방대한 양의 text 데이터로 진행. 이렇게 학습된 LLM은 next word를 더 잘 예측하고, fluent하고 문법적으로 올바르게 생성할 수 있게 됨.
    - But, 우리는 그냥 fluent한걸 원하는게 아니라 단어 연결 이상의 것을 원함 :
        1. Helpful : 유용하고 관련있는 정보 제공
        2. Harmless : 위험하거나 편향적인 정보 생성 피함
        3. Aligned with Human Preferences : 사람이 자연스럽고, 유용하고, 매력적으로 느끼는 방식으로 응답
- Pretraining LLM 
    - text data로부터 next word를 예측하는 것에 의존하는 방식
    - 위의 측면에서 부족함
- Supervised Training
    - 구조화된 output을 생성하는 것에 좋음
    - but, helpful,harmless, aligned response들을 생성하는 것에 덜 효과적
- Fine-tuned model
    - 조금 더 유창하고 구조적인 text를 생성
    - but, 여전히 incorrect, biased, 정답이 아닌 응답 등의 문제 존재
- RL을 도입
    - RL은 이러한 사전훈련된 LLM을 더 잘 finetune하기 위한 방법
    - 이를 통해 단순히 말을 잘하는게 아니라 실제 도움이 되도록 할 수 있음

### RLHF

- 매우 유명한 기법
    - RL에서 reward signal의 proxy로 human feedback을 사용
- 동작 방식
    1. Get Human Preferences
        - Same input에 대해 2가지 다른 답변을 보여주고 사용자가 더 나은 답변을 선택하도록 함.
    2. Train a Reward Model 
        - reward model이라 불리는 분리된 model을 훈련시키기 위해 위에서 얻은 데이터를 사용
        - 이 모델은 사람이 좋아할 것인 응답의 종료를 예측
            - +, helpfulness,harmlessness,alignment with human preferences에 근거하여 응답을 score하기 위해 학습
    3. Fine-tune the LLM with RL
        - 위에서 만든 reward model을 LLM의 environment로 사용.
        - 과정
            - LLM이 응답 생성
            - Reward model이 scoring
            - LLM이 reward model이 좋다고 생각하는 text를 생성하도록 훈련
        ![RL Diagram](https://huggingface.co/reasoning-course/images/resolve/main/grpo/2.jpg)

- 일반적인 관점에서 LLM에 RL을 사용하는것의 이점

| Benefit|Description|
|---|---|
|Improved Control|RL은 LLM이 생성하는 text의 종류를 넘어 더 잘 control할 수 있게 해줌. 더 구체적인 목표에 맞춰 text를 생성하도록 할 수 있음 |
| Enhanced Alignment with Human Values | 특히 RLHF는 LLM이 복잡하고 주관적인 human preference에 align 되도록 도움. 그냥 우리가 *좋은 응답을 작성하기 위한 rule을 작성하는건* 어렵지만, 사람이 선호하는 응답을 비교 선택하게 하는건 쉬움. |
| Mitigating Undesirable Behaviors| RL은 LLM에서 negative behavior를 줄이는데 사용될 수 있다.  잘못된 행동에 penalty를 주도록 reward를 design함으로써 모델이 그러한 행동을 피하도록 할 수 있다.|

- RLHF에는 많은 방식들이 있음 그 중 GRPO라는 기법을 다룬다.

### Why should we care about GRPO?

- GRPO가 llm에 대한 RL에서 상당한 발전을 보였기 떄문
- 다른 technique 2가지를 먼저 간단하게 봄
    1. PPO
    2. DPO

- [PPO (Proximal Policy Optimization)](https://huggingface.co/docs/trl/main/en/ppo_trainer)
    - RLHF에 대한 First effective techniques 중 하나
    - policy gradient method를 사용하여 policy를 업데이트
- [DPO (Direct Preference Optimization)](https://huggingface.co/docs/trl/main/en/dpo_trainer)
    - reward model에 대한 필요를 제거한 simpler technique으로 뒤에 개발되었음
    - reward model이 없으므로 directly data를 이용
    - DPO는 기본적으로 chosen과 rejected response 사이의 classification task로 문제를 구성
- DPO GPO랑 다르게 GRPO는 비슷한 sample들을 group으로 만들고 group으로 비교함
    - group based approcch의 장점
        - stable gradient
        - better convergence properties
- GRPO는 DPO랑 다르게 preference data를 사용하지 않고, reward signal을 사용하여 비슷한 sample들의 group들을 비교
- GRPO는 reward signal을 얻는 것에서 flexible함
    - PPO처럼 reward model을 사용함
    - 하지만, strictly하게 요구하지는 않음.
        - GRPO는 여러 function/model로부터 reward signal들을 incorporate할 수 있기 떄문
    - EX. 우리는 shorter reponse에 대해 보상하기 위해 length function을 사용할 수 있음
        - 또는 더 정확한 응답에 보상하기 위해 factual correctness function을 사용할 수 있음
    - 이러한 유연성이 GRPO가 부분적으로 변하게 쉽도록 함.

## Understanding the DeepSeek R1 Paper

- DeepSeek R1 : LM training에서 significant advancement를 보임
    - 특히 Reinforcement Learning을 이용하여 Reasoning 능력을 개발한 것
    - DeepSeek R1 발표 논문에서 GRPO(Group Relative Policy Optimization)이라는 new reinforcement learning algorithm을 발표
    - DeepSeek 논문의 initial goal은 순수 reinforcement learning이 SFT없이 Reasoning Capabilities를 개발할 수 있는지 탐구하는 것

- Next Chapter에서 논문 기반으로 GRPO를 구현 예정

### The Breakthrough 'Aha' Moment
- R1-Zero의 훈련에서 most remakable discoveries중 하나는 `Aha moment`
    - `Aha moment`는 사람이 실제로 문제 해결과정에서 갑자기 깨닫게 될 때와 유사
- Model의 동작 과정
    1. Initial Attempt : 모델이 문제 해결에서 첫 시도를 시작
    2. Recognition : 잠재적 error, inconsistencies를 인지
    3. Self-Correction : 2에서 인식한 내용을 바탕으로 접근 방식 수정
    4. Explanation : 왜 새로운 접근이 더 나은지 설명할 수 있음

## Implementing GRPO in TRL

- 이번 챕터에서는 **TRL**을 사용해서 **GRPO를 구현**하는 방법에 대해 다룸

- Remind some of important concepts of GRPO Algorithm
    1. Group Formation : Model은 각 prompt에 대해 multiple completion 생성
    2. Preference Learning : 모델은 group들을 비교하는 reward function으로 학습
    3. Training Configuration : Model은 training process를 control하기 위해 configuration 사용

- GRPO 구현을 위해 해야하는거
    1. Define a dataset of prompts
    2. Define a reward function : completions list를 받아서 reward list를 반환하는 reward function
    3. GRPOConfig로 training process configure
    4. GRPOTrainer를 사용하여 model 훈련

- GRPO Training 예시

``` python
from trl import GRPOTrainer, GRPOConfig
from datasets import load_dataset

# 1. Load your dataset
dataset = load_dataset("your_dataset", split="train")


# 2. Define a simple reward function
def reward_func(completions, **kwargs):
    """Example: Reward longer completions"""
    return [float(len(completion)) for completion in completions]


# 3. Configure training
training_args = GRPOConfig(
    output_dir="output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    logging_steps=10,
)

# 4. Initialize and train
trainer = GRPOTrainer(
    model="your_model",  # e.g. "Qwen/Qwen2-0.5B-Instruct"
    args=training_args,
    train_dataset=dataset,
    reward_funcs=reward_func,
)
trainer.train()
```

- 위 코드의 Key Component
1. Dataset Format
    - Dataset은 model이 응답할 prompt를 포함해야 함.
    - GRPO trainer는 각 prompt에 대해 multiple completion을 생성하고, reward function을 사용해서 결과들을 비교

2. Reward Function
    - 이 보상 함수가 중요함. 모델이 어떻게 학습할 것인지 결정하는 함수

``` python
# Example 1: Reward based on completion length
def reward_length(completions, **kwargs):
    return [float(len(completion)) for completion in completions]


# Example 2: Reward based on matching a pattern
import re

def reward_format(completions, **kwargs):
    pattern = r"^<think>.*?</think><answer>.*?</answer>$"
    return [1.0 if re.match(pattern, c) else 0.0 for c in completions]
```

3. Training Configuration

``` python
training_args = GRPOConfig(
    # Essential parameters
    output_dir="output",
    num_train_epochs=3,
    num_generation=4,  # Number of completions to generate for each prompt
    per_device_train_batch_size=4,  # We want to get all generations in one device batch
    # Optional but useful
    gradient_accumulation_steps=2,
    learning_rate=1e-5,
    logging_steps=10,
    # GRPO specific (optional)
    use_vllm=True,  # Speed up generation
)
```

- `num_generation` : GRPO에서 특히 중요 (group size 결정)(각 prompt 별로 답변을 몇 개나 생성할까 결정)

    - Too small (eg. 2-3) : 충분히 다양하지 않음(의미 있는 비교가 안 되는 양)
    - Recommended(4-16) : **diversity와 computational efficiency의 적절한 balance 제공**
    - Larger values : 학습을 improve 하겠지만, computational cost가 많이 증가

- Group Size는 내 computational resource에 근거해서 선택해야 함. 간단한 작업이면 4-8 정도면 충분한데, 복잡한 추론 작업이면 8-16정도의 큰 group이 필요

### Tips for Success

1. Memory Management : `per_device_train_batch_size`와 `gradient_accumulation_steps`를 내 GPU memory에 맞게 조정
2. Speed : `use_vllm=True`로 설정하면 내 모델이 지원된다면 faster generation 가능
3. Monitoring : training동안 metric 기록
    - `reward` : completion 전체의 평균 reward
    - `reward_std` : reward group의 standard deviation(표준편차)
    - `kl` : reference model과의 KL divergence

### Reward Function Design

- Deepseek R1 paper에서 몇가지 효과적인 reward function design을 보여줌

1.  Length-Based Rewards
    - 구현하기 쉬운 보상 중 하나는 length-based reward 긴 응답에 reward 줄 수 있음
    - 아래와 같이 코드를 작성하면 내가 원하는 길이보다 너무 길거나 짧은 응답에 penalty를 줄 수 있음
    
``` python
def reward_len(completions, **kwargs):
    ideal_length = 20
    return [-abs(ideal_length - len(completion)) for completion in completions]
```

2. Rule-Based Rewards for Verifiable Tasks
    - 객관적인 정답이 있는 문제에 대해서 아래처럼 rule-based reward function을 작성할 수도 있다.

``` python
def problem_reward(completions, answers, **kwargs):
    """Reward function for math problems with verifiable answers
    completions: list of completions to evaluate
    answers: list of answers to the problems from the dataset
    """

    rewards = []
    for completion, correct_answer in zip(completions, answers):
        # Extract the answer from the completion
        try:
            # This is a simplified example - you'd need proper parsing
            answer = extract_final_answer(completion)
            # Binary reward: 1 for correct, 0 for incorrect
            reward = 1.0 if answer == correct_answer else 0.0
            rewards.append(reward)
        except:
            # If we can't parse an answer, give a low reward
            rewards.append(0.0)

    return rewards
```

3. Format-Based Rewards
    - DeepSeek R1 training에서 중요했던 proper formatting에 reward 줄 수 있음.

``` python
def format_reward(completions, **kwargs):
    """Reward completions that follow the desired format"""
    # Example: Check if the completion follows a think-then-answer format
    pattern = r"<think>(.*?)</think>\s*<answer>(.*?)</answer>"

    rewards = []
    for completion in completions:
        match = re.search(pattern, completion, re.DOTALL)
        if match:
            # Check if there's substantial content in both sections
            think_content = match.group(1).strip()
            answer_content = match.group(2).strip()

            if len(think_content) > 20 and len(answer_content) > 0:
                rewards.append(1.0)
            else:
                rewards.append(
                    0.5
                )  # Partial reward for correct format but limited content
        else:
            rewards.append(0.0)  # No reward for incorrect format

    return rewards
```

## Practical Exercise: Fine-tune a model with GRPO

- 실제 GRPO를 이용하여 모델을 finetuning하는 것을 해볼 예정
    
### Install dependencies

``` bash
!pip install -qqq datasets==3.2.0 transformers==4.47.1 trl==0.14.0 peft==0.14.0 accelerate==1.2.1 bitsandbytes==0.45.2 wandb==0.19.7 --progress-bar off
!pip install -qqq flash-attn --no-build-isolation --progress-bar off
```

### Import Libraries

``` bash
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
```

### Import and log in to Weights & Biases

- Weight & bias는 실험 monitoring, logging tool
- finetuning process 기록에 사용

``` python
import wandb

wandb.login()
```

### Load the dataset

- 본인이 finetuning에 사용할 dataset을 고르면 된다.
    - 예시에서는 short stories list가 들어있는 [mlabonne/smoltldr](https://huggingface.co/datasets/mlabonne/smoltldr) dataset 사용

``` python
dataset = load_dataset("mlabonne/smoltldr")
print(dataset)
```

### Load Model

- 이제 실제 훈련시킬 모델을 가져오기
    - 예제에선,  [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) model 사용
        - GPU가 보통 제한적이라서 작은 모델 선택한 것
- 모델 GET

``` python
model_id = "HuggingFaceTB/SmolLM-135M-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
    attn_implementation="flash_attention_2",
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
```

### Load LoRA

- 이제 LoRA configuration을 가져오고, Trainable parameter 숫자를 줄이기 위해 LoRA를 사용할 예정

- [LoRA](https://kingrang-e.tistory.com/4)는 이전 내용에서 다루었음 

``` python
# Load LoRA
lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=16,
    lora_alpha=32,
    target_modules="all-linear",
)
model = get_peft_model(model, lora_config)
print(model.print_trainable_parameters())
```

### Define the reward function

- 이전 Chapter에서 말했듯, GRPO는 model 향상을 위해 어떠한 reward function도 사용할 수 있음.
    - 이번 예시에선, 이전 단락에서 말한 length관련 simple reward function을 사용할 예정

``` python
ideal_length = 50


def reward_len(completions, **kwargs):
    return [-abs(ideal_length - len(completion)) for completion in completions]
```

### Define the training arguments

- 이제 훈련을 위해 training arguments를 정의해야 함.
    - `GRPOConfig`를 사용하여 training argument를 전형적인 `transformer` style로 정의할 예정
    - Training Argument에 대한 정보가 필요하면 [여기](https://huggingface.co/docs/transformers/en/main_classes/trainer#trainingarguments) 확인

``` python
# Training arguments
training_args = GRPOConfig(
    output_dir="GRPO", # 결과 저장 폴더
    learning_rate=2e-5, # learning rate
    per_device_train_batch_size=8, # batch size
    gradient_accumulation_steps=2, # gradient 누적 step (2이므로 batch 16과 같은 효과)
    max_prompt_length=512, # 최대 prompt 길이
    max_completion_length=96, # 최대 완료 길이
    num_generations=8, # 생성할 샘플 수
    optim="adamw_8bit", # optimizer
    num_train_epochs=1, # epoch
    bf16=True, # datatype
    report_to=["wandb"], #. using wandb to monitor
    remove_unused_columns=False, 
    logging_steps=1, 
)
```

- 이제 우리는 model, dataset, training argument로 시작할 수 있다.

``` python
# Trainer
trainer = GRPOTrainer(
    model=model,
    reward_funcs=[reward_len],
    args=training_args,
    train_dataset=dataset["train"],
)

# Train model
wandb.init(project="GRPO")
trainer.train()
```

### Push the model to the Hub during training

- 만약 우리가 `push_to_hub` argument를 True로 하고 `model_id` argument를 유효한 모델 이름으로 했다면, 훈련 하는 동안 Model은 HuggingFace Hub에 업로드 된다.
    - vibe testing을 바로 시작하는 경우에 유용함

### Interpret training results

- `GRPOTrainer`는 reward function의  reward, loss, 다른 metric 범위로 log
    - 우리는 **reward**와 **loss**만에 초점
- reward의 경우 아래 그림처럼 훈련할수록 0에 가까워지고, 이건 모델이 잘 훈련되었다는 것

![](https://huggingface.co/reasoning-course/images/resolve/main/grpo/13.png)


- loss의 경우에는 **0에서 시작해서 점점 loss가 증가하는데** 이는 직관적이지 않게 보인다. 이 행동은 **GRPO에서 예상되고 algorithm의 수학적인 공식과 직접적 연관이 있음.**    
    - GRPO에서 loss는 KL divergence(original policy 대비 상한액)과 비례. training이 진행되는 동안, 모델은 reward function과 더 잘 match되는 text를 생성하기 위해 학습하고, 이로인해 초기 policy로부터 더 벗어나게 됨. 
    - 이렇게 loss가 증가하는 것은 실제로 reward function에 잘 맞춰 훈련되고 있다는 것을 나타냄

![](https://huggingface.co/reasoning-course/images/resolve/main/grpo/14.png)

### Save and publish the model

- model을 community에 공유

``` python
merged_model = trainer.model.merge_and_unload()
merged_model.push_to_hub(
    "SmolGRPO-135M", private=False, tags=["GRPO", "Reasoning-Course"]
)
```

### Generate text

- 이제 Finetuning이 성공적으로 완료되었으므로, model로 text를 생성해보자

``` python
prompt = """
# A long document about the Cat

The cat (Felis catus), also referred to as the domestic cat or house cat, is a small 
domesticated carnivorous mammal. It is the only domesticated species of the family Felidae.
Advances in archaeology and genetics have shown that the domestication of the cat occurred
in the Near East around 7500 BC. It is commonly kept as a pet and farm cat, but also ranges
freely as a feral cat avoiding human contact. It is valued by humans for companionship and
its ability to kill vermin. Its retractable claws are adapted to killing small prey species
such as mice and rats. It has a strong, flexible body, quick reflexes, and sharp teeth,
and its night vision and sense of smell are well developed. It is a social species,
but a solitary hunter and a crepuscular predator. Cat communication includes
vocalizations—including meowing, purring, trilling, hissing, growling, and grunting—as
well as body language. It can hear sounds too faint or too high in frequency for human ears,
such as those made by small mammals. It secretes and perceives pheromones.
"""

messages = [
    {"role": "user", "content": prompt},
]
```

- 위와같이 Prompt를 작성하고, 이제 model로 text를 생성할 수 있다.

``` python
# Generate text
from transformers import pipeline

generator = pipeline("text-generation", model="SmolGRPO-135M")

## Or use the model and tokenizer we defined earlier
# generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

generate_kwargs = {
    "max_new_tokens": 256,
    "do_sample": True,
    "temperature": 0.5,
    "min_p": 0.1,
}

generated_text = generator(messages, generate_kwargs=generate_kwargs)

print(generated_text)
```