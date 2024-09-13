from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.prompts import PromptTemplate

prompt = PromptTemplate.from_template("what is {word} ? ")

llm = HuggingFacePipeline.from_model_id(
    model_id="gpt2",
    task="text-generation",
)

chain = prompt | llm

out = chain.invoke({"word": "tomato"})
print(out)