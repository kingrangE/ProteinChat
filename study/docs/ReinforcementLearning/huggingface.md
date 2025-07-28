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

- 이 과정은 단순한 암기보다는 학습을 보여준다. 아래에서 예시를 통해 Aha moment를 갖는 순간을 상상하자
    - EX, 퍼즐을 해결하는 상황
        1. First try : 이 조각은 색을 보니까 여기 둬야해
        2. Recognition : 잠만, 모양 안 맞네
        3. Self-Correction : 아 그럼 저기에 둬야한다.
        4. Explanation : 왜냐면 이 위치에서 색, 모양 둘다 맞잖아

- 이 능력은 단순한 훈련 데이터 암기가 아니라 RL로 학습하는 것

### The Training Process
- R1 훈련은 multi-phase process
    - 각 phase와 phase의 key innovation 분석
        - final process는 2가지 모델 생성
            1. DeepSeek-R1-Zero : 순서하게 RL만으로 훈련된 모델
            2. DeepSeek-R1 : DeepSeek-R1-Zero을 기반으로 SFT하여 만든 모델
            | Feature | DeepSeep-R1-Zero | DeepSeek-R1 |
            | ---|---|---|
            | Training Approach | Pure RL | Multi-Phase (SFT+RL)|
            | FineTuning | None | Supervised FineTuning |
            | Reasoning Capability | Emergent | Enhanced | 
            | Key Characteristics | Strong reasoning but readability issues | Better language consistency and readability |
    - DeepSeek-R1-Zero가 순수 RL의 잠재성을 증명하고, DeepSeek-R1이 Reasoning과 usability를 모두 우선시하는 balanced approach를 통해 기반을 다짐

- 트레이닝은 총 4단계로 구분된다.
    1. Cold Start Phase
    2. Reasoning RL Phase
    3. Rejection Sampling Phase
    4. Diverse RL Phase

1. Cold Start Phase
    ![Cold Start Explanation in Paper](https://huggingface.co/reasoning-course/images/resolve/main/grpo/5.png)

    - Cold Start 단계는 model의 읽기 능력과 응답 품질을 위해 strong foundation을 다지도록 design되었음
    - R1-Zero으로 생성한 high quality 소규모 데이터셋을 사용하여 V3-Base Model을 Finetuning
    - 해당 innovative approach는 small high quality dataset을 사용하여 baseline readability and response quality를 확립
2. Reasoning RL Phase
    ![Reasoning RL in Paper](https://huggingface.co/reasoning-course/images/resolve/main/grpo/6.png)
    - RL phase는 core reasoning capabilities 개발에 집중
        - 해당 단계는 rule-based reinforcement learning을 사용
    - 중요 ! : RL 단계의 모든 task는 `verifiable`하기 때문에 모델의 정답이 맞았는지 틀렸는지 체크할 수 있다.
        - EX, 수학의 경우, mathematical solver를 이용해서 모델의 정답이 맞았는지 틀렸는지 확인 가능
    - 이 단계를 특히 innovative하게 만드는 것은 direct optimization approach
        - 이는 separate reward model의 필요를 없애고 training process를 간소화함
3. Rejection Sampling Phase (Quality Control)
    ![Rejection Sampling Phase in Paper](https://huggingface.co/reasoning-course/images/resolve/main/grpo/7.png)
    - Rejection Sampling Phase동안 Model은 sample을 생성하고 quality control process를 통해 filter
    - DeepSeek V3는 pure reasoning task를 넘어 broad scope로 quality jude로서 evaluating output을 제공
    - Filtering된 데이터는 SFT에 이용된다.
    - 이 단계의 innovation은 **high standard output을 보장하기 위해 multiple quality signal을 조합하기 위한 능력에 있음**
4. Diverse RL Phase (Broad Alignment)
    ![Diverse RL Phase](https://huggingface.co/reasoning-course/images/resolve/main/grpo/8.png)
    - 마지막 Diverse RL Phase에서는 **sophisticated hybrid approach**를 통해 multiple task type을 다룸
        1. Deterministic Task
            - rule-based reward를 사용
        2. Subjective Task
            - LLM Feedback을 통해 평가
    - 해당 단계는 Hybrid Approach를 통해 Human Preference Alignment를 달성하는 것
        - Hybrid Approach : flexibility of language model evaluation과 rule-based system의 정확도를 조합

### The Algorithm: Group Relative Policy Optimization (GRPO)
- 이제 모델 훈련에 사용하는 algorithm 확인하기
    - 논문 저자들은 GRPO를 model finetuning의 돌파구로서 설명함
- GPRO의 novelty는 **directly optimize for preference rectification**을 하기 위한 능력에 있음
    -  이는 model이 우리가 원하는 output을 내도록 Align하기 위한 direct and efficient route를 의미 (PPO같은 전통적 알고리즘이랑 대조되는)
- GRPO가 어떻게 동작하는지 3가지 main component들을 통해 알아보자
    1. Group Formation : Creating Multiple Solutions
        ![First Step of GRPO](https://huggingface.co/reasoning-course/images/resolve/main/grpo/11.jpg)
        - GRPO에서 첫 단계는 직관적
            - 이는 어떻게 학생들이 어려운 문제를 여러 접근을 시도함으로써 푸는지와 비슷함
            1. Prompt가 주어지면, 모델은 그냥 응답을 생성하는게 아니라 **문제를 풀기 위한 multiple attempts를 한다.** (4,8,16)
            2. 모든 이러한 시도들은 group으로 모아서 유지된다.
                - multiple student들의 solution을 비교하고 학습할 수 있는 것처럼
    2. Preference Learning: Understanding What Makes a Good Solution
        ![](https://huggingface.co/reasoning-course/images/resolve/main/grpo/12.jpg)
        - GRPO는 간단함이 진짜 GOOD (really shine in its simplicity)
            - RLHF는 solution의 평가를 위해 separate reward model이 요구되지만, **GRPO는 어떠한 function이든 model이든 사용할 수 있다.**
        - Evaluation Process는 다양한 면에서 각 응답을 본다.
            1. 마지막 정답이 맞아?
            2. 정답이 proper formatting을 따르고 있어?
            3. 추론이 제공된 답변과 일치해?
        - 이 접근 방식을 특별하게 만드는 것은 **어떻게 점수를 다루는가**
            - GRPO는 reward를 각 group안에서 normalize (not just giving score)
                - simple but effective fomular사용
            ``` python
            Advantage = (reward - mean(group_rewards)) / std(group_rewards)
            ```
        - 이 normalization은 curve에서 grading하는 것과 같음
            - 이는 모델이 group안에서 어떤 응답이 더 좋고 나쁜지 이해하는데 도움이 된다.
    3. Optimization: Learning from Experience
        - 마지막 단계는 GRPO가 각 그룹의 solution을 평가하여 학습한 것으로부터 모델을 improve하는 방법을 가르치는 내용
            - 이 단계는 powerful and stable 모두 가짐 (2가지 단계로 구성)
                1. 모델이 less effective approach로부터 벗어나 **successful solution을 만들도록 함**
                2. 모델이 한 번에 크게 변하는 것을 막는 **safety mechanism(KL divergence penalty)을 포함**함
        - 이러한 접근은 traditional method보다 더 stable하다.
            1. 한 번에 두 개를 비교하는게 아니라 **여러 개를 한 번에 보기 때문**
            2. group based normalization은 **reward scaling 문제를 방지하는데 도움을 주기 때문**
            3. KL penalty이 **safety net같이 행동**하고 **model이 새로운 것을 학습하는 동안에 이미 아는 것을 까먹지 않도록 보장하기 때문**

### GRPO Algorithm in Pseudocode
``` text
Input: 
- initial_policy: Starting model to be trained
- reward_function: Function that evaluates outputs
- training_prompts: Set of training examples
- group_size: Number of outputs per prompt (typically 4-16)

Algorithm GRPO:
1. For each training iteration:
   a. Set reference_policy = initial_policy (snapshot current policy)
   b. For each prompt in batch:
      i. Generate group_size different outputs using initial_policy
      ii. Compute rewards for each output using reward_function
      iii. Normalize rewards within group:
           normalized_advantage = (reward - mean(rewards)) / std(rewards)
      iv. Update policy by maximizing the clipped ratio:
          min(prob_ratio * normalized_advantage, 
              clip(prob_ratio, 1-epsilon, 1+epsilon) * normalized_advantage)
          - kl_weight * KL(initial_policy || reference_policy)
          
          where prob_ratio is current_prob / reference_prob

Output: Optimized policy model
```

### Limitations and Challenges of GRPO
- significant advancement를 보여주었지만, limitation과 challenge를 이해하는 것이 중요
    1. Generation Cost : 한 prompt에 대해 여러 개의 응답을 생성하는 것은 계싼량을 증가시킨다. 
    2. Batch Size Constraint : groups of completion을 모두 처리하기 위한 필요성으로 인해 effective batch size에 제한될 수 있고, 학습 process가 복잡해질 수 있고, 잠재적으로 훈련이 오래걸릴 수 있음
    3. Reward Function Design : Training의 품질은 reward function에 심하게 의존.
    4. Group Size Tradeoffs : optimal group size를 선택하는 것은 solution diversity와 computational cost의 밸런스가 중요. 
    5. KL Divergence Tuning : KL divergence penalty에 대한 올바른 균형을 찾는 것은 careful tuning을 요구함
        - 너무 높으면 모델이 효과적으로 학습 X (변화가 적어짐), 낮으면 initial capabilities에서 많이 벗어남 (변화가 큼)