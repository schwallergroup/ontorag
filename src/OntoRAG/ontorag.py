"""Generate synthesis procedures with ontology-based RAG."""

from typing import Tuple

import dspy
from .utils import OntoRetriever
from dsp.utils import deduplicate
from typing import Optional, List, Dict
import json


class OntoRAG(dspy.Module):
    def __init__(self, num_passages=5, context=None, ontology_path: Optional[str] = None):
        """Initialize the module with a document retriever, an ontology querier, and a predictor.

        Args:
            num_passages: Number of passages to retrieve from the database.
            context: Optional, passing the ground truth context.
        """

        super().__init__()
        self.retriever = dspy.Retrieve(k=num_passages)
        self.ontoretriever = OntoRetriever(ontology_path = ontology_path)

        self.context = context
        self.predictor = None

    def forward(self, query: str) -> Tuple:
        """Main loop of ontology-based RAG."""
        pass

    def retrieve(self, query: str):
        ctxt_doc, ctxt_onto = "", ""

        if self.context is None:
            ctxt_dict = self.retrieve_doc(query)
            ctxt_doc = self.format_context(ctxt_dict)

        if self.ontoretriever.ontology.ontologies:
            ctxt_onto = self.ontoretriever(query)
            ctxt_onto = self.format_onto_context(ctxt_onto)

        ctxt = self.fuse_contexts(ctxt_doc, ctxt_onto)
        return ctxt

    def format_context(self, context: List[Dict]) -> str:
        """Format context."""
        contexts = [p["text"] for c in context for p in c["passages"]]
        return "\n".join(deduplicate(contexts))
    
    def format_onto_context(self, context: List[Dict]) -> str:
        """Format ontology context."""
        return json.dumps(context, indent=2)
    
    def fuse_contexts(self, ctxt_doc: str, ctxt_onto: str) -> str:
        """Fuse document and ontology contexts."""
        return ctxt_doc + ctxt_onto





