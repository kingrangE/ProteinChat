# 전길원 Huggingface LLM Course 
- [Huggingface Learn LLM Course](https://huggingface.co/learn/llm-course/chapter1/1?fw=pt)를 토대로 공부한 내용을 정리하였습니다.

# 1 Transformer Models
## Introduction
### Understanding NLP and LLMs
-  NLP와 LLMs의 차이
    - NLP 
        - computer가 human language를 이해하고, 해석하고 생성할 수 있게 하는 것에 focus하는 broader field
        - many techniques, tasks를 포괄함
            - ex, 감정 분석, name entity recognition, machine translation etc.
    - LLMs
        - NLP model의 powerful subset
        - 많은 데이터, massive한 size, minimal task-specific training으로 wide range task를 수행할 수 있는 기능을 특징으로 함
        - Llama, GPT, Claude 등이 존재함

## NLP and LLMs
### What is NLP?
- NLP는 언어, 그리고 인간 언어와 관련된 모든 것을 이해하는 것에 focus한 ML분야이다.
- 또한, NLP의 목표는 개별 언어 이해 뿐만 아니라, 이러한 언어들의 context를 이해하는 것이다.
- 일반적인 NLP 분야는 아래와 같다.
    - 전체 문장 분류 : spam mail 분류, 문장 옳고 그름 여부 확인 등
    - 문장 내부의 단어 분류 : 문장의 문법적인 component와 named entitie들을 확인하는 것
    - text content 생성
    - text로부터 정답 추출
    - input text로부터 새로운 문장 생성

- NLP는 단순히 글을 쓰는거에만 한정되지 않는다. TTS or Image Generation과 같은 Speech recognition, computer vision과도 관련이 있음. 

### LLM의 부상
- GPT, Llama가 출시되며 language processing 분야에서 더 많은 것이 가능해짐.
- LLM의 특징
    - Scale : LLM은 파라미터 수가 최소 million~ 최대 trillion까지 규모가 매우 큼
    - General Capabilities : LLM은 task-specific training없이 다양한 task를 수행할 수 있음
        - NLP의 경우 한 가지 task를 정해 훈련
    - In-Context learning : LLM은 prompt에 있는 몇 가지 예시로 훈련 가능
    - Emergent abilities : model의 크기가 커지면서, 프로그래밍되거나 예상되지 않는 기능을 보여줌

- LLM의 등장으로 기존 specialized model 만들기에서 prompt, fine-tuned으로 wide range of language task를 해결하는 쪽으로 paradigm이 변화했다.

- 하지만 LLM은 아래와 같은 주요 한계 또한 갖는다.
    1. Hallucination
    2. true understanding 부족 : 통계적 패턴에 따라서만 동작하고, 실제 세계에 대한 이해는 부족함.
    3. bias : training data, input에 대해 편향을 재생산함.
    4. Context window : 개선되고 있긴 하지만, 제한적인 context window를 가짐.
    5. Computational Resoureces : 많은 양의 computational resource를 요구함.

- 왜 language processing이 어려울까?
    - 길게 적혀있지만 결론은 사람이 이해하는 것처럼 machine이 이해할 수 없기 때문. 이는 LLM의 발전에서도 여전히 남아있음

## Transformers, What can we do?
- 이 단락에서는 Transformer model이 할 수 있는 것과 pipeline function에 대해서 얘기함
### Transformers are everywhere!
- Transformer model은 모든 유형의 문제를 해결하는데 사용된다. NLP, CV, Audio Processinsssg, etc.

### Working with pipelines
- Transformer library의 basic concept == **pipeline**
    - pipeline은 model을 necessary preprocessing과 postprocessing step과 함께 연결하고, 우리가 text를 넣으면 intelligible answer를 얻을 수 있게 해줌.
- pipeline example
    ```python
    from transformers import pipeline

    classifier = pipeline("sentiment-analysis")
    classifier("I've been waiting for a HuggingFace course my whole life.")
    ```
    ```python
    #결과
    [{'label': 'POSITIVE', 'score': 0.9598047137260437}]
    ```
    - classifier에 문자열이 아닌, list를 전달해도 각 문장에 대한 결과가 잘 출력된다.