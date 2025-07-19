# 전길원 Soft Prompt 공부 - Huggingface Soft Prompt
- 출처 : [Huggingface Docs PEFT SoftPrompt](https://huggingface.co/docs/peft/conceptual_guides/prompting)

- LLM을 훈련하는 것은 시간이 많이 들고, 계산량이 많은 작업
    - Model의 사이즈가 커지면서, prompting과 같은 Efficient Training Method에 대한 관심이 증가하는 중
- Prompting 이란?
    - **task에 대한 예시를 설명하거나 task를 설명하는 prompt를 포함하는 specific downstream task에 대해 frozen pretrained model을 준비하는 것**
- Prompting에서 각 task에 대해 분리해서 fully training을 하지 않고, **동일한 frozen pretrained model을 사용한다.**
    - 이건 다른 작업들에 동일한 모델을 사용할 수 있기 때문에 쉽고, 모든 모델의 params를 훈련하는 것보다 작은 prompt params set을 훈련하고 저장하기에 훨씬 효율적이다.

- 2가지 prompting method
    1. Hard Prompt
        - Discrete Input Token이 있는 Handcrafted Text Prompt를 사용하는 방식
        - 좋은 프롬프트를 만들기 위해 수작업으로 하기에 매우 힘듦
    2. Soft Prompt
        - Dataset에 최적화될 수 있는 input embedding을 가진 Learnable Tensors
        - 단점은 사람이 읽을 수 없다는 것 (실수들의 나열이기 때문)

- 이 내용에서는 soft prompt method에 대한 brief overview를 제공
    - prompt tuning, prefix tuning, P-tuning, multitask prompt tuning

## Prompt tuning
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

### Prompt Tuning for Causal Language Modeling - 코드
- [HuggingFace Docs Prompt Tuning](https://huggingface.co/docs/peft/main/en/task_guides/clm-prompt-tuning)

## Prefix tuning
![Prefix Tuning](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/peft/prefix-tuning.png)
- 각 Task에 대해 prefix parameter들을 optimize

- Prefix Tuning은 GPT 모델의 NLG task에 맞춰 design되었음
- Prompt Tuning과 매우 유사한데, prefix tuning 또한 task-specific vector sequence를 input sequence앞에 붙이는 방식이다. 또한 앞에 붙이는 vector는 훈련 가능하고, 업데이트 가능하다.
- 차이점은?
    - prefix parameters는 모든 model layer에 삽입되지만, prompt tuning은 오직 model input embedding에만 추가된다.
    - 또한, prefix parameter는 soft prompt에서 직접적으로 training되는 대신에 별도의 FFN에 의해 최적화된다. (instability and hurts performance 때문)
        - FFN은 soft prompt update 후에 버려짐

- 결과적으로 1000배 이상 적은 파라미터를 훈련함에도 분리하고 좋은 성능을 보임

### Prefix Tuning for conditional generation - 코드
- [HuggingFace Docs Prefix Tuning](https://huggingface.co/docs/peft/main/en/task_guides/seq2seq-prefix-tuning)

## P-tuning
![P-tuning](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/peft/p-tuning.png)
- Prompt token들은 input sequence 어디든 삽입될 수 있고, prompt encoder에 의해 최적화된다.
- P-tuning은 모든 언어 모델과 NLU(Natural Language Understanding)task에 맞춰 design되었다. 
    - soft prompt method의 another variation
        - P-tuning 또한 better prompt를 찾기 위해 최적화될 수 있는 trainable embedding tensor를 추가하고, prompt parameter를 최적화하기 위해 prompt encoder를 사용한다.
- But, prefix tuning과는 다름
    1. prompt token들은 input sequence 어디에든 삽입될 수 있음. (prefix는 앞부분으로 위치가 제한되어 있음)
    2. prompt token들은 모델의 모든 레이어에 추가되는 대신 오직 input에만 추가된다.
    3. anchor token을 도입하는 것은 performance를 향상시킬수 있다. (input sequence 구성요소의 특징을 나타내기 때문)
- P-tuning은 prompt를 수동으로 만드는 것보다 효율적이고, GPT 같은 모델(Decoder 모델)이 NLU 작업에서 BERT(Encoder 모델)같은 모델과 경쟁이 가능해지게 함

### P-tuning for sequence classification
- [Huggingface Docs P-tunin](https://huggingface.co/docs/peft/main/en/task_guides/ptuning-seq-classification)

## Multitask prompt tuning
- [Multitask prompt tuning](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/peft/mpt.png)
- MPT는 different target task에 대해 공유될 수 있는 multiple task types data로부터 single prompt를 학습한다.
    - 기존의 방식 : target task 적응을 위해 retrieve/aggregate될 필요가 있는 **각 task에 대해 별도의 soft prompt를 학습**
- MPT는 2 stage로 구성:
    1. source training
        - 각 task의 soft prompt는 task-specific vector로 분해. 
        - 분해된 task specific vector들은 서로 곱해져서 또 다른 행렬 $W$를 구성하게 되고, task-specific prompt matrix를 생성하기 위해 $W$와 shared prompt matrix $P$ 사이에서 Hadamard product가 사용된다. 
        - 생성된 task-specific prompts는 모든 task에 걸쳐 공유되는 single prompt matrix로 distill되고 이 prompt는 multitask training으로 훈련된다.
    2. target adaption
        - target task에 대해 single prompt를 adapt하기 위해, target prompt는 초기화되고 shared prompt matrix와 task-specific low-rank prompt matrix의 Hadamard product로 표현된다.
    ![Prompt Decomposition](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/peft/mpt-decomposition.png)

## Context-Aware Prompt Tuning (CPT)
![CPT](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/peft/cpt.png)
- CPT는 specific token embedding만 최적화 (model의 나머지 부분은 frozen)
- CPT는 오직 context embedding을 refine함으로써 few shot classification을 향상시키도록 design되어있다.
    - 이건 In-Context Learning과 Prompt Tuning, Adversarial Optimization으로부터 아이디어를 결합하여 model adaption을 parameter efficient and effective 하게 만드는데 중점을 둔 방식.
    - CPT에서는 specific context token embedding만 최적화되고 모델의 나머지 부분은 frozen
- overfitting을 막고, stability를 유지하기 위해서 CPT는  controlled perturbation을 사용하여 정의된 범위 내에서 context embedding에 변경 사항을 제한한다.
    - controlled perturbation?

- 추가적으로 recency bias 현상을 설명하기 위해서 CPT는 decay loss factor를 적용함
    - recency bais 
        - 모델의 끝 부분에 있는 example이 앞 부분에 있는 것보다 우선시 되는 현상

### CPT Code 
[CPT finetuning github](https://github.com/huggingface/peft/blob/main/examples/cpt_finetuning/README.md)