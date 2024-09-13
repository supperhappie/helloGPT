from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate

prompt = PromptTemplate.from_template("Wht is {word}?")
llm = HuggingFaceHub(
    repo_id = "google/gemma-2-2b-it",
    # repo_id = "mistralai/Mistral-7B-Instruct-v0.1",
    task="text-generation",
    # model_kwargs={
    #     "max_new_tokens":250,
    # },
)
chain=prompt|llm
llm.client.api_url = 'https://api-inference.huggingface.co/models/google/gemma-2-2b-it'
# llm.client.api_url = 'https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1'
out = chain.invoke({"word":"food"})
print(out)