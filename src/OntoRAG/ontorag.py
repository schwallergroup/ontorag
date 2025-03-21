"""Generate synthesis procedures with ontology-based RAG."""

import json
from typing import Dict, List, Optional, Tuple, Union

import dspy
from dspy.dsp.utils import deduplicate

from .utils import OntoRetriever


class BaseOntoRAG(dspy.Module):
    retriever: dspy.Retrieve
    ontoretriever: OntoRetriever

    def forward(self, query: str) -> dspy.Signature:
        """Forward pass of the OntoRAG pipeline."""
        pass

    def retrieve(self, query: str, ctxt_doc: Optional[str] = None) -> str:
        """Retrieve and format."""
        ctxt_doc, ctxt_onto = "", ""

        if ctxt_doc is None:
            ctxt_dict = self.retrieve_doc(query)
            ctxt_doc = self.format_context(ctxt_dict)

        if self.ontoretriever.ontology.ontologies:
            ctxt_ontoj = self.ontoretriever(query)
            ctxt_onto = self.format_onto_context(ctxt_ontoj)

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
