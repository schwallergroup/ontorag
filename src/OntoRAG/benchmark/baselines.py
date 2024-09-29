"""Baseline methods for biomed Q&A benchmarks."""

from dotenv import load_dotenv
import dspy
from typing import Tuple, Optional

__all__ = ['QAZeroShot', 'QAContext', 'QAReason', 'QAFull']

qprompt = "Here is the question you need to answer:"
choice_prompt = "Answer to the question. Only one character."
context_prompt = "Here is the context:"
reasoning_prompt = "Before answering the question, carefully analyze the ontology context. Finalize by selecting the correct answer."

class MedQnA_ZeroShot(dspy.Signature):
    """Answer a question with a detailed response."""
    question: str = dspy.InputField(desc=qprompt)
    choice_answer: str = dspy.OutputField(desc=choice_prompt)

class MedQnA_Context(dspy.Signature):
    """Answer a question with a detailed response."""
    context: str = dspy.InputField(desc=context_prompt)
    question: str = dspy.InputField(desc=qprompt)
    choice_answer: str = dspy.OutputField(desc=choice_prompt)

class MedQnA_Reason(dspy.Signature):
    """Answer a question with a detailed response."""
    question: str = dspy.InputField(desc=qprompt)
    reasoning: str = dspy.OutputField(desc=reasoning_prompt)
    choice_answer: str = dspy.OutputField(desc=choice_prompt)

class MedQnA_Full(dspy.Signature):
    """Answer a question with a detailed response."""
    context: str = dspy.InputField(desc=context_prompt)
    question: str = dspy.InputField(desc=qprompt)
    reasoning: str = dspy.OutputField(desc=reasoning_prompt)
    choice_answer: str = dspy.OutputField(desc=choice_prompt)


# Modules
class QAZeroShot(dspy.Module):
    """Simply ask a question and get an answer."""
    def __init__(self, ontology_path: Optional[str] = None, context: Optional[str] = None):
        super().__init__()
        self.predict = dspy.Predict(MedQnA_ZeroShot)

    def forward(self, query: str) -> Tuple[MedQnA_ZeroShot, str]:
        answer = self.predict(question=query)
        return answer, None

class QAContext(dspy.Module):
    """Ask question, retrieve context and directly get answer."""
    def __init__(self, ontology_path: Optional[str] = None, context: Optional[str] = None):
        super().__init__()
        self.predictor = dspy.Predict(MedQnA_Context)
        self.retriever = dspy.Retriever()

    def forward(self, query: str) -> Tuple[MedQnA_Context, str]:
        context = self.retrieve(query)
        answer = self.predictor(question=query, context=context)
        return answer, context

class QAReason(dspy.Module):
    """Ask question, get reasoning and answer."""
    def __init__(self, ontology_path: Optional[str] = None, context: Optional[str] = None):
        super().__init__()
        self.predictor = dspy.Predict(MedQnA_Reason)

    def forward(self, query: str) -> Tuple[MedQnA_Reason, str]:
        answer = self.predictor(question=query)
        return answer, None

class QAFull(dspy.Module):
    """Ask question, quert context, get reasoning and answer."""
    def __init__(self, ontology_path: Optional[str] = None, context: Optional[str] = None):
        super().__init__()
        self.predictor = dspy.Predict(MedQnA_Full)
        self.retriever = dspy.Retriever()

    def forward(self, query: str) -> Tuple[MedQnA_Full, str]:
        context = self.retrieve(query)
        answer = self.predictor(question=query, context=context)
        return answer, context


if __name__ == '__main__':
    load_dotenv()
    llm = dspy.OpenAI(
        system_prompt="",
        model='gpt-4o-mini',
        max_tokens=254,
    )
    dspy.settings.configure(lm=llm)

    orag = QAZeroShot(context='')
    print(orag.forward('What kinds of health care encounters exist?'))