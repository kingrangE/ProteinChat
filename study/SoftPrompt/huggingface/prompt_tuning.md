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
- Setup
