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
    - 우리가 위에서 pipeline을 create할때 model은 download되고, cache된다.
        - cache된 모델은 다음 번 사용해서 다시 download할 필요가 없어짐
- 내가 pipeline으로 text를 보낼 때, 아래와 같은 three main step이 진행된다.
    1. 모델이 이해할 수 있는 형태로 text가 preprocess된다.
    2. model로 preprocess된 text가 전달된다.
    3. model의 예측은 post process되어 우리가 이해할 수 있는 형태가 된다.

- different modalities를 위해 사용할 수 있는 pipeline
    - pipeline function은 여러 modalities(text,image,audio, mutlimodal task 등)를 지원한다.
    - 여기에서는 text tasks만 focus하지만, 실제로는 다양한 것들이 있고 [링크](https://huggingface.co/docs/hub/en/models-tasks)에서 확인 가능하다.
- kind of text pipeline
    1. text-generation
    2. text-classification
    3. summerization
    4. translation
    5. zero-shot-classification
    6. feature-extraction
- kind of image pipeline
    1. image-classification
    2. image-segmentation
    3. object-detection
- kind of audio pipeline
    1. audio-speeck-recognition
    2. audio-classification
    3. text-to-speech
- Multimodal pipeline
    - image-text-to-text
- Pipeline Example
    - Basic
        ```python
        from transformers import pipeline

        generator = pipeline("text-generation")
        generator("In this course, we will teach you how to")

        """결과
        [{'generated_text': 'In this course, we will teach you how to understand and use '
                    'data flow and data interchange when handling user data. We '
                    'will be working with one or more of the most commonly used '
                    'data flows — data flows of various types, as seen by the '
                    'HTTP'}]
        """
        ```
    - If i want to use any specific model
        ```python
        from transformers import pipeline

        generator = pipeline("text-generation", model="HuggingFaceTB/SmolLM2-360M")
        generator(
            "In this course, we will teach you how to",
            max_length=30,
            num_return_sequences=2,
        )
        """결과
        [{'generated_text': 'In this course, we will teach you how to manipulate the world and '
                    'move your mental and physical capabilities to your advantage.'},
        {'generated_text': 'In this course, we will teach you how to become an expert and '
                            'practice realtime, and with a hands on experience on both real '
                            'time and real'}]
        """
        ```
### Combining data from multiple sources
- Transformer model의 powerful application은 다양한 source로부터 데이터를 결합하고 처리하는 능력이다.
    - 특히 필요한 순간
        1. multiple database or repositories에서 검색
        2. different format(audio,text,image)들로부터 information 통합
        3. related information의 unified view 생성
    - 예를 들어, 아래 같은 시스템 제작 가능
        1. text, image와 같은 multiple modalities에서 정보 Search
        2. different source를 하나의 coherent response로 결합 -> audio file과 text description
        3. 많은 docs, metadata로부터 가장 관련된 정보를 보여줌

## How do Transformers work?
### Transformers are language models
- 모든 Transformer model은 language models로 훈련되어 왔다. 이것은 곧 model들이 많은 양의 raw text로 self-supervised 훈련되었다는 것
    - self-supervised learning : model의 input으로부터 objective를 자동적으로 계산하는 훈련 방식, 이로써 사람들은 데이터에 라벨링할 필요가 없어졌다.
### General Transformer architecture
- Transformer model은 2개의 block으로 구성되어 있음
    ![Transformer](https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/transformers_blocks-dark.svg)
    - Encoder : input을 받고, input의 representation을 build
        - 이는 model이 input으로부터 understanding을 얻도록 최적화 되어있음을 의미함
    - Decoder : encoder's representation과 다른 input을 사용하여 target sequence를 생성.
        - 이는 model이 output을 생성하는데 최적화되어 있음을 의미
- 이러한 파트 각 부분은 작업에 따라 **독립적으로 사용**될 수 있다.
    - **Encoding-only models** : input 이해를 요구하는 task에 좋다. (sentence classification, named entity recognition)
    - **Decoding-only models** : 생성하는 task에 좋다. (text generation)
    - **Encoder-Decoder models** or **sequence-to-sequence models** : input을 요구하는 generative task에 좋음 (translation, summarization)
- 더 자세한 Architecture 내용은 후반부에서 다시 다룸

### Attention layers
- Transformer model의 key feature는 **attention layer**
- Attention layer의 상세한 내용은 후반부에서 다시 다룸
    - 우선 여기서 알아야 하는 것은 Attention layer의 역할
        - Attention layer는 각 단어의 표현을 처리할 때, 전달한 문장의 특정 단어에 주의를 기울이도록 하고 다른 단어는 무시하도록 한다.
        - EX) Eng -> French translation task
            - I like this course를 번역한다고 가정하면
            - model은 like에 대한 적절한 번역을 하기 위해 I에도 Attention해야함.(동사가 주어에 따라 달라져서) 
            - 그러나 그 외에는 like 번역에 중요하지 않음. 이런 것처럼 this를 해석한다고 하면, this는 course에 집중해야 함.
        - 이 컨셉은 어떤 task에도 적용된다.
            - 단어는 그 자체로 의미를 갖지만, context에 의해 Deeply affected 받기 때문

### The original architecture
- ![Transformer original architecture](https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/transformers-dark.svg)
- Transformer Architecture는 기본적으로 translation을 위해 design되었다.
    - Training동안, encoder는 특정 언어의 input을 받는 반면에, decoder는 요구되는 target 언어의 같은 문장을 받는다. Encoder에서 Attention layer는 문장안의 모든 단어를 사용하나, Decoder는 sequentially하게 동작하고 이미 번역된 문장의 단어에만 pay attention한다.
        - EX, 첫 세단어 translate가 된 상태라면 다음 단계에선 네번째 단어 예측을 위해 encoder의 모든 input을 함께 사용하여 예측함.
    - Decoder block에서 첫번째 attention layer는 모든 past input에 대해 attention이다. 하지만, 두번째 attention layer는 encoder의 output을 사용하므로 전체 입력 문장에 대해 접근하여 더 잘 예측할 수 있고 이걸 통해서 더 어려운 언어에서도 유용하게 사용할 수 있다.(문법적으로 순서가 바뀌는 그런 언어 번역도 해결 가능)
    - Attention mask는 encoder/decoder에서 특정 special word에 집중하는 것을 막기 위해 사용될 수 있다. 
        - EX) 문장을 다같이 batching할 때, special padding word는 모든 input을 같은 길이로 만들기 위해 사용한다.

### Architecture와 checkpoint
- Architecture : model의 뼈대 - 각 layer와 모델 안에서 일어나는 각 동작의 정의
- Checkpoints : 주어진 Architecture에서 load될 weights
- Model : Upbrella term, 모든 의미를 포괄적으로 가짐

