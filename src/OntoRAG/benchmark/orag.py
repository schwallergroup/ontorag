"""An OntoRAG for biomedical Q&A."""

from OntoRAG.ontorag import BaseOntoRAG
from OntoRAG.utils import OntoRetriever
from dotenv import load_dotenv
import dspy
from typing import Literal, Tuple, Optional

__all__ = ['SimpleORAG', 'HyQORAG']

class MedQnA(dspy.Signature):
    """Answer a question with a detailed response based on the given context."""
    context: str = dspy.InputField(desc="Here is the ontology context")
    question: str = dspy.InputField(desc="Here is the question you need to answer")
    reasoning: str = dspy.OutputField(desc="Before answering the question, carefully analyze the ontology context. Finalize by selecting the correct answer.")
    choice_answer: str = dspy.OutputField(desc="Answer to the question. Only one character.")


# Implement multiple methods/variations of OntoRAG

class SimpleORAG(BaseOntoRAG):
    """Identify and query concepts in question, then generate answer."""
    def __init__(self, ontology_path, context: Optional[str] = None):
        super().__init__()
        self.predictor = dspy.Predict(MedQnA)
        self.ontoretriever = OntoRetriever(ontology_path = ontology_path)

    def forward(self, query: str) -> Tuple[MedQnA, str]:
        context = self.retrieve(query)
        answer = self.predictor(question=query, context=context)
        return answer, context


class HyQORAG(BaseOntoRAG):
    """Generate hypothetical answer, then query concepts in answer and reconsider response."""
    def __init__(self, ontology_path, context: Optional[str] = None):
        super().__init__()
        self.hypot_answer = dspy.Predict(MedQnA)
        self.final_predictor = dspy.Predict(MedQnA)
        self.ontoretriever = OntoRetriever(ontology_path = ontology_path)

    def forward(self, query: str) -> Tuple[MedQnA, str]:
        # Generate hypothetical answer
        ctxt0 = self.retrieve(query)
        hans = self.hypot_answer(question=query, context=ctxt0)

        # Query concepts in hypothetical answer
        ctxt1 = self.retrieve(hans.reasoning + hans.choice_answer)
        answer = self.final_predictor(question=query, context=ctxt1)

        return answer, ctxt1


if __name__ == '__main__':
    load_dotenv()
    llm = dspy.OpenAI(
        system_prompt="",
        model='gpt-4o-mini',
        max_tokens=254,
    )
    dspy.settings.configure(lm=llm)

    orag = SimpleORAG(ontology_path='data/test/ontologies/SNOMED', context='')
    print(orag.forward('What kinds of health care encounters exist?'))