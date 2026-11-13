from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate


def build_langchain_chain(prompt_template, llm):
    if llm is None:
        return None
    return prompt_template | llm | StrOutputParser()


def run_chain(chain, **kwargs):
    if chain is None:
        return None
    return chain.invoke(kwargs)


def build_prompt_template(template, variables):
    return PromptTemplate(template=template, input_variables=variables)
