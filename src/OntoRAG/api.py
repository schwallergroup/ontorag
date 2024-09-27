# -*- coding: utf-8 -*-

"""Main code."""

from OntoRAG.ontorag import BaseOntoRAG
from OntoRAG.utils import OntoRetriever
from dotenv import load_dotenv
import dspy


class QnA(dspy.Signature):
    """Answer a question with a detailed response based on the given context."""
    context: str = dspy.InputField(desc="Here is the ontology context")
    question: str = dspy.InputField(desc="Here is the question you need to answer")
    reasoning: str = dspy.OutputField(desc="Before answering the question, carefully analyze the ontology context.")
    answer: str = dspy.OutputField(desc="Answer to the question.")


class OntoRAG(BaseOntoRAG):
    def __init__(self, ontology_path, context):
        super().__init__()
        self.predictor = dspy.Predict(QnA)
        self.ontoretriever = OntoRetriever(ontology_path = ontology_path)

    def forward(self, query):
        context = self.retrieve(query)
        print(context)
        answer = self.predictor(question=query, context=context)
        return answer


if __name__ == '__main__':
    load_dotenv()
    llm = dspy.OpenAI(
        system_prompt="",
        model='gpt-4o-mini',
        max_tokens=254,
    )
    dspy.settings.configure(lm=llm)

    orag = OntoRAG(ontology_path='data/test/ontologies/SNOMED', context='')
    print(orag.forward('What kinds of health care encounter exist?'))