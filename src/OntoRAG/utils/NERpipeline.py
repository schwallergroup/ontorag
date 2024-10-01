import json
import os
from collections import defaultdict

import owlready2
import spacy
from spacy.matcher import PhraseMatcher


class OntologyNER:
    def __init__(self, ontology_folder, debug=False):
        self.ontology_folder = ontology_folder
        self.debug = debug
        self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        self.ontologies = self.load_ontologies()
        self.combined_matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        self.concept_to_ontology = {}
        self.create_combined_matcher()
        self.lineage_cache = defaultdict(dict)

    def process_statement(self, statement):
        """Retrieve ontology concepts from a statement."""
        recognized_concepts = self.recognize_concepts(statement)

        output = {}
        for onto_name, concepts in recognized_concepts.items():
            output[onto_name] = {"concepts": concepts, "terms": {}}
            for concept in concepts:
                lineage = self.get_concept_lineage(concept, onto_name)
                if lineage:
                    output[onto_name]["terms"][concept] = lineage

        return output

    def load_ontologies(self):
        """Load all ontologies from the specified folder."""
        ontologies = {}
        for filename in os.listdir(self.ontology_folder):
            if filename.endswith(".owl"):
                onto_name = filename[:-4]
                onto_path = os.path.join(self.ontology_folder, filename)
                onto = owlready2.get_ontology(onto_path).load()
                ontologies[onto_name] = {
                    "ontology": onto,
                    "properties": self.get_properties(onto),
                }
                if self.debug:
                    print(f"Loaded ontology: {onto_name}")
        return ontologies

    def create_combined_matcher(self):
        """Create a matcher for all ontologies."""
        for onto_name, ontops in self.ontologies.items():
            onto, props = ontops["ontology"], ontops["properties"]
            patterns = []
            for entity in onto.classes():
                if entity.label:
                    for label in entity.label:
                        concept = label.lower()
                        patterns.append(self.nlp(concept))
                        self.concept_to_ontology[concept] = onto_name
                        # Add singular/plural variations
                        if concept.endswith("s"):
                            singular = concept[:-1]
                            patterns.append(self.nlp(singular))
                            self.concept_to_ontology[singular] = onto_name
                        else:
                            plural = concept + "s"
                            patterns.append(self.nlp(plural))
                            self.concept_to_ontology[plural] = onto_name
            self.combined_matcher.add(onto_name, patterns)
            if self.debug:
                print(
                    f"Added {len(patterns)} patterns for ontology: {onto_name}"
                )

    def recognize_concepts(self, text):
        """Recognize ontology concepts in a text."""
        doc = self.nlp(text.lower())
        matches = self.combined_matcher(doc)

        recognized_concepts = defaultdict(set)
        for match_id, start, end in matches:
            onto_name = self.nlp.vocab.strings[match_id]
            concept = doc[start:end].text
            recognized_concepts[onto_name].add(concept)

        if self.debug:
            print(f"Recognized concepts: {dict(recognized_concepts)}")
            print(f"Tokens: {[token.text for token in doc]}")
            print(f"Matcher results: {matches}")

        return {k: list(v) for k, v in recognized_concepts.items()}

    def get_properties(self, onto):
        """Retrieve the definition of a concept in an ontology."""

        def get_first_value(annotation):
            if annotation:
                value = annotation.first()
                return str(value) if value is not None else "Not available"
            return "Not available"

        def collect_props_cls(cls):
            label_property = "http://www.w3.org/2000/01/rdf-schema#label"
            definition_property = "http://purl.obolibrary.org/obo/IAO_0000115"
            comment_property = "http://purl.obolibrary.org/obo/IAO_0000116"

            label = get_first_value(cls.label)
            definition = get_first_value(cls.IAO_0000115)
            comment = get_first_value(cls.IAO_0000116)
            return {
                "label": label,
                "definition": definition,
                "comment": comment,
            }

        properties = {}
        try:
            for cls in onto.classes():
                prps = collect_props_cls(cls)
                properties[prps["label"]] = prps
            return properties
        except:
            return {}

    def get_concept_lineage(self, concept, onto_name):
        """Retrieve the lineage of a concept in an ontology."""
        if concept in self.lineage_cache[onto_name]:
            return self.lineage_cache[onto_name][concept]

        onto = self.ontologies[onto_name]["ontology"]
        prps = self.ontologies[onto_name]["properties"]
        cls = onto.search_one(label=concept)

        if not cls:
            if self.debug:
                print(
                    f"Could not find class for concept: {concept} in ontology: {onto_name}"
                )
            return None

        superclasses = [
            str(c.label.first())
            for c in cls.ancestors()
            if c != cls
            if c.label.first() is not None
        ]
        subclasses = [
            str(c.label.first())
            for c in cls.subclasses()
            if c.label.first() is not None
        ]

        lineage = {
            **prps.get(concept, {}),
            "parents": superclasses,
            "children": subclasses,
        }
        self.lineage_cache[onto_name][concept] = lineage
        return lineage

    def print_ontology_concepts(self, onto_name):
        """Print all concepts in an ontology."""
        if onto_name not in self.ontologies:
            print(f"Ontology {onto_name} not found.")
            return
        onto = self.ontologies[onto_name]
        print(f"Concepts in {onto_name}:")
        for cls in onto.classes():
            if cls.label:
                print(f"  - {cls.label}")


if __name__ == "__main__":
    oret = OntologyNER(
        ontology_folder="data/ontologies/sacs_claude3_5", debug=True
        # ontology_folder="data/test/ontologies/SNOMED", debug=True
    )
    q = oret.process_statement("what is photocatalysts")
    print(json.dumps(q, indent=2))
