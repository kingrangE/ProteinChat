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