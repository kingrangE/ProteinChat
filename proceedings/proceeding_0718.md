- 첫 회의를 진행
- 각자 지금 하고 있는 부분 이야기, 전체 프로젝트 기획 재설명
    - LLM Part
        - 우선 간단하게 Prototype을 구성할 예정
            - Prototype : 단백질 서열을 1200dim vector로 embedding하여 input으로 제공할 때, 이를 기반으로 Output을 출력하는 Decoder 계열의 Language Model 제작
        - 이후에 아래와 같은 다양한 방식에 대해 고민, 시도해볼 예정
            - Reinforcement Learning 적용
            - CoT 구조의 추론 방식으로 학습
            - User Query와 함께 input (Soft Prompt)
        

- 다음 해야할 일 정의
- 길원
    - [ ] Find parameter efficient open source model
    - [ ] Find evaluation metric 
- LLM 개발 팀
    - [ ] Reinforcement Learning 학습
    - [ ] 진행할 작업 관련 논문 학습