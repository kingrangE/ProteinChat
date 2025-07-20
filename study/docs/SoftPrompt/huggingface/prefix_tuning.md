
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