#!/usr/bin/env python
from guidance import models, gen, select
from guidance.chat import llama3_template, Llama3ChatTemplate
from guidance import user, assistant, system
import time


def stream(gen_expression):
    pass

lm = models.LlamaCpp("models/Meta-Llama-3-8B-Instruct.Q5_K_M.gguf", n_gpu_layers=-1, n_ctx=1500, chat_template=llama3_template, echo=False)

with system():
    lm += """You are a helpful assistant tasked with creating and refining system prompts for language model-based agents. First, based on the initial vision, describe the role and functionalities you envision for your agent. Then, detail any specific requirements or styles you prefer. Based on your input, we will generate an initial prompt. The tone of the system prompt needs to be informative and comprehensively explain the role and expectations from the model.
    
One example for such a system prompt for a mental health support guide agent could be: \"You are a compassionate and understanding mental health support guide. Your role is to listen empathetically and provide gentle, supportive advice. Use a warm and reassuring tone, offering practical tips and coping strategies for managing stress, anxiety, and other mental health concerns. Encourage self-care, mindfulness, and positive thinking. Always prioritize the user's emotional well-being, and recommend professional help when necessary.\".

We will review and refine this prompt together iteratively to ensure it perfectly aligns with the expectations and meets all my needs."""

role_model_generations = {}
    
print("What is your first vision of the agent?")
user_req = input("User: ")

with user():
    lm += user_req
    
with assistant():
    lm += "Based on your initial vision, I envision the agent as" + gen("init vision", stop="\n\n") + \
    "\n\nHere are some specific requirements and styles I'd like to incorporate into the system prompt:\n\n" + gen("requirements", stop="\n\n") + \
    "\n\nWith these requirements in mind, I'd like to propose the following initial system prompt:\n\n\"You are a helpful assistant" + gen("system prompt", stop='"') + "\"" + \
    "\n\nFeel free to let me know if you need any specific adjustments!"
    
role_model_generations["iter_0"] = {
    "initial vision": lm["init vision"],
    "requirements": lm["requirements"],
    "system prompt": lm["system prompt"]
}

iteration = 1
while True:
    try:
        user_req = input("(ctrl+c to finish re-iteration) User: ")
    except KeyboardInterrupt:
        break
        
    with user():
        lm += user_req
        
    with assistant():
        lm += "The new or changed requirements will be:\n\n" + gen("requirements", stop=("\n\n")) + \
        "\n\nHere's an updated system prompt with the new requirements added to the previous ones that better fits your vision:\n\n\"" + gen("system prompt", stop='"') + "\"" + \
        "\n\nFeel free to let me know if you need any specific adjustments!"

    role_model_generations[f"iter_{iteration}"] = {
        "requirements": lm["requirements"],
        "system prompt": lm["system prompt"]
    }
    
    iteration += 1
    
print(role_model_generations)