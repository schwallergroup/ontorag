"""Baseline methods for biomed Q&A benchmarks."""

from typing import Literal, Optional, Tuple

import dspy
from dotenv import load_dotenv
from utils import qa_decorator

__all__ = ["QAZeroShot", "QAContext", "QACoT", "RAGCoT", "QATwoStep"]

qprompt = "Here is the question you need to answer:"
choice_prompt = "Answer: ${answer} (yes/no)"
context_prompt = "Here is the context:"
reasoning_prompt = (
    "Reasoning: Let's think step by step in order to ${reasoning}"
)


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


class BaseQA(dspy.Module):
    def forward(self, qprompt: str) -> dspy.Signature:
        """Forward pass of a RAG pipeline."""
        pass


# Modules
class QAZeroShot(BaseQA):
    """Simply ask a question and get an answer."""

    def __init__(
        self, ontology: Optional[str] = None, context: Optional[str] = None
    ):
        super().__init__()
        self.predict = dspy.Predict(MedQnA_ZeroShot)

    # docstr-coverage:inherited
    @qa_decorator()
    def forward(self, qprompt: str) -> dspy.Prediction:
        answer = self.predict(question=qprompt)
        answer.context = None
        return answer


class QAContext(BaseQA):
    """Ask question, retrieve context and directly get answer."""

    def __init__(
        self, ontology: Optional[str] = None, context: Optional[str] = None
    ):
        super().__init__()
        self.predictor = dspy.Predict(MedQnA_Context)
        self.rm = dspy.Retrieve()

    # docstr-coverage:inherited
    @qa_decorator()
    def forward(self, qprompt: str) -> dspy.Prediction:
        context = self.retrieve(qprompt)
        answer = self.predictor(question=qprompt, context=context)
        answer.context = context
        return answer

    def retrieve(self, qprompt: str) -> str:
        """Update this with real retrieval."""
        passages = self.rm(qprompt).passages
        return "\n".join(passages)


class QACoT(BaseQA):
    """Ask question, get reasoning and answer."""

    def __init__(
        self, ontology: Optional[str] = None, context: Optional[str] = None
    ):
        super().__init__()
        self.predictor = dspy.Predict(MedQnA_Reason)

    # docstr-coverage:inherited
    @qa_decorator(reasoning=True)
    def forward(self, qprompt: str) -> dspy.Prediction:
        answer = self.predictor(question=qprompt)
        answer.context = None
        return answer


class RAGCoT(BaseQA):
    """Ask question, query context, get reasoning and answer."""

    def __init__(
        self, ontology: Optional[str] = None, context: Optional[str] = None
    ):
        super().__init__()
        self.predictor = dspy.Predict(MedQnA_Full)
        self.rm = dspy.Retriever()

    # docstr-coverage:inherited
    @qa_decorator(reasoning=True)
    def forward(self, qprompt: str) -> dspy.Prediction:
        context = self.retrieve(qprompt)
        answer = self.predictor(question=qprompt, context=context)
        answer.context = context
        return answer

    def retrieve(self, qprompt: str) -> str:
        """Update this with real retrieval."""
        passages = self.rm(qprompt).passages
        return "\n".join(passages)


# TODO Implement a twostep method with no ontology context, for control
class QATwoStep(BaseQA):
    """Ask question, get reasoning and answer in two steps."""

    def __init__(
        self, ontology: Optional[str] = None, context: Optional[str] = None
    ):
        super().__init__()
        self.predictor = dspy.Predict(MedQnA_Reason)

    # docstr-coverage:inherited
    @qa_decorator(reasoning=True)
    def forward(self, qprompt: str) -> dspy.Prediction:
        answer0 = self.predictor(question=qprompt)
        answer = self.predictor(
            question=answer0.reasoning + answer0.choice_answer
        )
        answer.context = None
        return answer


if __name__ == "__main__":
    load_dotenv()
    llm = dspy.OpenAI(
        system_prompt="",
        model="openai/gpt-4o-mini",
        max_tokens=254,
    )
    dspy.settings.configure(lm=llm)

    orag = QAZeroShot(context="")
    print(
        orag.forward(
            "Thymoquinone is ineffective against radiation induced enteritis, yes or no?. Is this true? (yes/no)"
        )
    )

    print(llm.history)
