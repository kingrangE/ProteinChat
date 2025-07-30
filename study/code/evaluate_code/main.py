import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from langchain_core.prompts import ChatPromptTemplate

ds = load_dataset("sequential-lab/TP53_protein_variants")

sequence = ds["train"]["sequence"][0]
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
    "sequence" : sequence
})


tokenizer = AutoTokenizer.from_pretrained("internlm/internlm-7b", trust_remote_code=True)

# Set `torch_dtype=torch.float16` to load model in float16, otherwise it will be loaded as float32 and might cause OOM Error.
model = AutoModelForCausalLM.from_pretrained("internlm/internlm-7b", torch_dtype=torch.float16, trust_remote_code=True).cuda()
model = model.eval()



inputs = tokenizer([prompt], return_tensors="pt")


for k,v in inputs.items():
    inputs[k] = v.cuda()

gen_kwargs = {"max_length": 5, 
              "temperature": 0.8, 
              "do_sample": True, 
              }

output = model.generate(**inputs, **gen_kwargs)
output = tokenizer.decode(output[0].tolist(), skip_special_tokens=True)

print(output)

# <s> A beautiful flower box made of white rose wood. It is a perfect gift for weddings, birthdays and anniversaries.
# All the roses are from our farm Roses Flanders. Therefor you know that these flowers last much longer than those in store or online!</s>
