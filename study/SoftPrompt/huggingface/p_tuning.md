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