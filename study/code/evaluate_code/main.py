import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from langchain_core.prompts import ChatPromptTemplate
import re
from tqdm import tqdm

def get_prompt(sequence):

    system_template = """
        You are a bioinformatics expert specializing in analyzing amino acid sequences to predict pathogenic variants.

        Classify the given amino acid sequence as '1 (pathogenic)' if it is likely to cause disease, or '0 (non-pathogenic)' if it is not.
        Your answer must be either '1' or '0'.

        Refer to the examples below to answer the final question.

        - example 1 (pathogenic)
        sequence: MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTA+
        label: 1

        - example 2 (non-pathogenic)
        sequence: MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCSDSDGLAPPQHLIRVEGNLRVEYLDDRNTFRHSVVVPYEPPEVGSDCTTIHYNYMCNSSCMGGMNRRPILTIITLEDSSGNLLGRNSFEVRVCACPGRDRRTEEENLRKKGEPHHELPPGSTKRALPNNTSSSPQPKKKPLDGEYFTLQIRGRERFEMFRELNEALELKDAQAGKEPGGSRAHSSHLKSKKGQSTSRHKKLMFKTEGPDSD
        label: 0
        """

    human_template = """
        - question 
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
    return prompt.to_string()


def evalute_model(dataset, model, tokenizer):
    correct = 0
    for sequence,answer in tqdm(zip(dataset["sequence"],dataset["label"])):
        prompt = get_prompt(sequence=sequence)
        inputs = tokenizer([prompt], return_tensors="pt")

        for k,v in inputs.items():
            inputs[k] = v.cuda()

        gen_kwargs = {"max_length": inputs.input_ids.shape[1]+5, 
                    "temperature": 0.8, 
                    "do_sample": True, 
                    }

        output = model.generate(**inputs, **gen_kwargs,use_cache=False)
        result = tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        print("출력 :",result)
        match = re.search(r'\d+', result)
        if match:
            try:
                predicted_answer = int(match.group(0))
                print("답 :",predicted_answer)
                if predicted_answer == answer:
                    correct += 1
            except (ValueError, IndexError):
                continue
    print("성능 :",correct/len(dataset))
        
ds = load_dataset("sequential-lab/TP53_protein_variants")
dataset = ds["train"]

tokenizer = AutoTokenizer.from_pretrained("internlm/internlm-7b", trust_remote_code=True)

# Set `torch_dtype=torch.float16` to load model in float16, otherwise it will be loaded as float32 and might cause OOM Error.
model = AutoModelForCausalLM.from_pretrained("internlm/internlm-7b", torch_dtype=torch.float16, trust_remote_code=True).cuda()
model = model.eval()


evalute_model(dataset,model,tokenizer)

# <s> A beautiful flower box made of white rose wood. It is a perfect gift for weddings, birthdays and anniversaries.
# All the roses are from our farm Roses Flanders. Therefor you know that these flowers last much longer than those in store or online!</s>
