"""Ontology retrieval module."""

import json
from typing import Dict, List, Optional

import dspy

from .NERpipeline import OntologyNER


class OntoRetriever(dspy.Module):
    """Ontology retrieval module."""

    def __init__(self, ontology_path: str):
        """Initialize the module with an ontology retriever.

        Args:
            ontology: Path to a directory with ontologies.
        """
        super().__init__()
        self.fpath = ontology_path
        self.ontology = OntologyNER(ontology_folder=ontology_path)

    def forward(self, query: str, context: Optional[str] = None) -> List[Dict]:
        """Retrieve ontology context."""

        onto_json = self.ontology.process_statement(query)
        return onto_json
