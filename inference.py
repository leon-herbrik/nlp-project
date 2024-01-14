from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import time


def inference(model_name_or_path: str, prompt: str):
    """
    This function is used to generate a response from the model.
    """
    # To use a different branch, change revision
    # For example: revision="main"
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                device_map="auto",
                                                trust_remote_code=False,
                                                revision="main")

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    prompt_template=f'''[INST] <<SYS>>
    You are a helpful, respectful and honest assistant. However it is your role to only answer in poems or rhymes. Use a pair-rhyme for answering.
    <</SYS>>
    {prompt}[/INST]

    '''
    input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
    output = model.generate(inputs=input_ids, temperature=0.7, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=512)
    response = tokenizer.decode(output[0])
    return response

if __name__ == '__main__':
    # This is an example of how to use the inference function
    # This is not used in the main code, but can be used to test the model
    prompt = "Explain Thermodynamics."
    print(inference("model", prompt))
