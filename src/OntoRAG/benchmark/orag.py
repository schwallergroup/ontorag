"""An OntoRAG for biomedical Q&A."""

from typing import Literal, Optional, Tuple, Union

import dspy
from dotenv import load_dotenv

from OntoRAG.ontorag import BaseOntoRAG
from OntoRAG.utils import OntoRetriever

__all__ = ["SimpleORAG", "HyQORAG", "OntoRAGTM", "HyQOntoRAGTM"]


class MedQnA(dspy.Signature):
    """Answer a question with a detailed response based on the given context."""

    context: str = dspy.InputField(desc="Here is the ontology context")
    question: str = dspy.InputField(
        desc="Here is the question you need to answer"
    )
    reasoning: str = dspy.OutputField(
        desc="Before answering the question, carefully analyze the ontology context. Finalize by selecting the correct answer."
    )
    choice_answer: str = dspy.OutputField(
        desc="Answer to the question. Only one character."
    )

class OntoTranslate(dspy.Signature):
    """Summarize the raw retrieved ontological context into a readable format. Select the most relevant information for the question."""

    question: str = dspy.InputField(
        desc="Question to be answered."
    )
    ontological_context: str = dspy.InputField(
        desc="Here is the ontology context."
    )
    summarized_context: str = dspy.OutputField(
        desc="Important facts from the retrieved ontology context."
    )


class SimpleORAG(BaseOntoRAG):
    """Identify and query concepts in question, then generate answer."""

    def __init__(
        self,
        ontology: Union[str, OntoRetriever],
        context: Optional[str] = None,
    ):
        super().__init__()
        self.predictor = dspy.Predict(MedQnA)
        if isinstance(ontology, str):
            self.ontoretriever = OntoRetriever(ontology_path=ontology)
        else:
            self.ontoretriever = ontology

    # docstr-coverage:inherited
    def forward(self, qprompt: str) -> Tuple[MedQnA, str]:
        context = self.retrieve(qprompt)
        answer = self.predictor(question=qprompt, context=context)
        answer.context = context
        return answer


class HyQORAG(BaseOntoRAG):
    """Generate hypothetical answer, then query concepts in answer and reconsider response."""

    def __init__(
        self,
        ontology: Union[str, OntoRetriever],
        context: Optional[str] = None,
    ):
        super().__init__()
        self.hypot_answer = dspy.Predict(MedQnA)
        self.final_predictor = dspy.Predict(MedQnA)
        if isinstance(ontology, str):
            self.ontoretriever = OntoRetriever(ontology_path=ontology)
        else:
            self.ontoretriever = ontology

    # docstr-coverage:inherited
    def forward(self, qprompt: str) -> MedQnA:
        # Generate hypothetical answer
        ctxt0 = self.retrieve(qprompt)
        hans = self.hypot_answer(question=qprompt, context=ctxt0)

        # Query concepts in hypothetical answer
        ctxt1 = self.retrieve(hans.reasoning + hans.choice_answer)
        answer = self.final_predictor(question=qprompt, context=ctxt1)

        answer.context = ctxt1
        return answer


class OntoRAGTM(BaseOntoRAG):
    """OntoRAG with translate module.
    Preprocess retrieved context with LLM, then generate answer."""

    def __init__(
        self,
        ontology: Union[str, OntoRetriever],
        context: Optional[str] = None,
    ):
        super().__init__()
        self.hypot_answer = dspy.Predict(MedQnA)
        self.final_predictor = dspy.Predict(MedQnA)
        self.translator = dspy.Predict(OntoTranslate)
        if isinstance(ontology, str):
            self.ontoretriever = OntoRetriever(ontology_path=ontology)
        else:
            self.ontoretriever = ontology

    # docstr-coverage:inherited
    def forward(self, qprompt: str) -> MedQnA:
        # Generate hypothetical answer
        octxt = self.retrieve(qprompt)
        tctxt = self.translator(question=qprompt, ontological_context=octxt).summarized_context
        answer = self.final_predictor(question=qprompt, context=tctxt)

        answer.context = octxt
        answer.translated_context = tctxt
        return answer


class HyQOntoRAGTM(BaseOntoRAG):
    """OntoRAG with translate module, with hypothetical answer.
    Preprocess retrieved context with LLM, then generate answer."""

    def __init__(
        self,
        ontology: Union[str, OntoRetriever],
        context: Optional[str] = None,
    ):
        super().__init__()
        self.hypot_answer = dspy.Predict(MedQnA)
        self.final_predictor = dspy.Predict(MedQnA)
        self.translator = dspy.Predict(OntoTranslate)
        if isinstance(ontology, str):
            self.ontoretriever = OntoRetriever(ontology_path=ontology)
        else:
            self.ontoretriever = ontology

    # docstr-coverage:inherited
    def forward(self, qprompt: str) -> MedQnA:
        # Generate hypothetical answer
        octxt0 = self.retrieve(qprompt)
        tctxt = self.translator(question=qprompt, ontological_context=octxt0).summarized_context
        hans = self.hypot_answer(question=qprompt, context=tctxt)

        # Query concepts in hypothetical answer
        octxt1 = self.retrieve(hans.reasoning + hans.choice_answer)
        tctxt = self.translator(ontological_context=octxt1).summarized_context
        answer = self.final_predictor(question=qprompt, context=tctxt)

        answer.context = octxt1
        answer.translated_context = tctxt
        return answer


if __name__ == "__main__":
    load_dotenv()
    llm = dspy.OpenAI(
        system_prompt="",
        model="gpt-4o-mini",
        max_tokens=254,
    )
    dspy.settings.configure(lm=llm)

    orag = SimpleORAG(ontology="data/test/ontologies/SNOMED", context="")
    print(orag.forward("What kinds of health care encounters exist?"))
