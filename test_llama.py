from llama_cpp import Llama
from guidance import models, gen, select
import time

llm = models.LlamaCpp("model/Meta-Llama-3-8B-Instruct.Q8_0.gguf")
# Timer start
start = time.time()
llm + f'Do you want a joke or a poem? ' + select(['joke', 'poem'])
print("Time taken: ", time.time() - start, " seconds")