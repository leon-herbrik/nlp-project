# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token="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")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token="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")