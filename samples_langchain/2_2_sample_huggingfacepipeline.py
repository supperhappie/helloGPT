from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.prompts import PromptTemplate

prompt = PromptTemplate.from_template("what is {word} ? ")

# this sample also connect to the huggingface
llm = HuggingFacePipeline.from_model_id(
    model_id="google/gemma-2-2b-it",
    task="text-generation",
    device = 0,
    pipeline_kwargs={"max_new_tokens":50},
)
# llm.client.api_url = 'https://api-inference.huggingface.co/models/google/gemma-2-2b-it'

chain = prompt | llm

out = chain.invoke({"word": "tomato"})

print(out)


# check torch and gpu recognition
import torch
print(torch.version.cuda)  # This will print the CUDA version PyTorch was built with
print(torch.backends.cudnn.enabled)  # Check if cuDNN (for deep learning) is enabled

print(torch.cuda.is_available())  # This should return True if a GPU is available
print(torch.cuda.device_count())  # This should return the number of available GPUs