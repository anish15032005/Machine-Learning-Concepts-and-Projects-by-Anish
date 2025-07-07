# from transformers.pipelines import pipeline
# import torch
# #check if GPU is available
# print(torch.cuda.is_available())
# print(torch.cuda.get_device_name(0))
# model = pipeline("summarization", model="facebook/bart-large-cnn", device=0)
# response = model("Text to summarize")
# print(response)

from transformers.pipelines import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers.utils.logging import set_verbosity_error

set_verbosity_error()

summarization_pipeline = pipeline("summarization", model="facebook/bart-large-cnn", device=0)
summarizer = HuggingFacePipeline(pipeline=summarization_pipeline)

refinement_pipeline = pipeline("summarization", model="facebook/bart-large", device=0)
refiner = HuggingFacePipeline(pipeline=refinement_pipeline)

qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2", device=0)

summary_template = PromptTemplate.from_template("Summarize the following text in a {length} way:\n\n{text}")

summarization_chain = summary_template | summarizer | refiner

text_to_summarize = input("\nEnter text to summarize:\n")
length = input("\nEnter the length (short/medium/long): ")

summary = summarization_chain.invoke({"text": text_to_summarize, "length": length})

print("\n🔹 **Generated Summary:**")
print(summary)

while True:
    question = input("\nAsk a question about the summary (or type 'exit' to stop):\n")
    if question.lower() == "exit":
        break

    qa_result = qa_pipeline(question=question, context=summary)

    print("\n🔹 **Answer:**")
    if isinstance(qa_result, dict) and "answer" in qa_result:
        print(qa_result["answer"])
    elif isinstance(qa_result, list) and len(qa_result) > 0 and "answer" in qa_result[0]:
        print(qa_result[0]["answer"])
    else:
        print("No answer found.")