# 모델 성능 평가용

- 병원성 변이(pathogenic variants) 가능성이 있는 단백질 서열인지 아닌지 판단하도록 제작
    - 데이터셋 : [TP53_protein_variants](https://huggingface.co/datasets/sequential-lab/TP53_protein_variants)

- Prompt
    - langchain ChatPromptTemplate 사용
    ``` python
    from langchain_core.prompts import ChatPromptTemplate
    system_template = """
        당신은 단백질의 아미노산 서열을 분석하여 병원성 변이(Pathogenic Variant) 여부를 예측하는 생물정보학 전문가입니다.

        주어진 아미노산 서열(sequence)이 질병을 유발할 가능성이 있는 '1(pathogenic)'인지, 그렇지 않은 '0(non-pathogenic)'인지 분류해주세요.
        답변은 반드시 '1' 또는 '0'으로만 해주세요.

        아래 예시를 참고하여 마지막 질문에 답변해주세요.

        # 예시 1 (병원성)
        - sequence: MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTA
        - label: 1

        # 예시 2 (비병원성)
        - sequence: MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCSDSDGLAPPQHLIRVEGNLRVEYLDDRNTFRHSVVVPYEPPEVGSDCTTIHYNYMCNSSCMGGMNRRPILTIITLEDSSGNLLGRNSFEVRVCACPGRDRRTEEENLRKKGEPHHELPPGSTKRALPNNTSSSPQPKKKPLDGEYFTLQIRGRERFEMFRELNEALELKDAQAGKEPGGSRAHSSHLKSKKGQSTSRHKKLMFKTEGPDSD
        - label: 0
        """
    human_template = """
        - sequence : {sequence}
        - label : 
    """

    prompt_template = ChatPromptTemplate({
        ("system",system_template),
        ("human",human_template)
    })

    prompt = prompt_template.invoke({
        "sequence" : "~~~~"
    })
    ```
- 평가 방식
    - 정확도 기반 평가 방식 
        - 정오답 비율
    