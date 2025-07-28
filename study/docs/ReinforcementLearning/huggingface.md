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

- 