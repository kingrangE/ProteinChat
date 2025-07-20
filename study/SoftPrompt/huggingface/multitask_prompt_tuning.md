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