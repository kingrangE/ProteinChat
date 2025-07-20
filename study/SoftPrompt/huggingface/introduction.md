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