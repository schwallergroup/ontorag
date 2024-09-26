import os
import json
import owlready2
from spacy.matcher import PhraseMatcher
import spacy
from collections import defaultdict

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
        recognized_concepts = self.recognize_concepts(statement)
        
        output = {}
        for onto_name, concepts in recognized_concepts.items():
            output[onto_name] = {
                "concepts": concepts,
                "terms": {}
            }
            for concept in concepts:
                lineage = self.get_concept_lineage(concept, onto_name)
                if lineage:
                    output[onto_name]["terms"][concept] = lineage

        return output

    def load_ontologies(self):
        ontologies = {}
        for filename in os.listdir(self.ontology_folder):
            if filename.endswith(".owl"):
                onto_name = filename[:-4]
                onto_path = os.path.join(self.ontology_folder, filename)
                onto = owlready2.get_ontology(onto_path).load()
                ontologies[onto_name] = onto
                if self.debug:
                    print(f"Loaded ontology: {onto_name}")
        return ontologies

    def create_combined_matcher(self):
        for onto_name, onto in self.ontologies.items():
            patterns = []
            for entity in onto.classes():
                if entity.label:
                    for label in entity.label:
                        concept = label.lower()
                        patterns.append(self.nlp(concept))
                        self.concept_to_ontology[concept] = onto_name
                        # Add singular/plural variations
                        if concept.endswith('s'):
                            singular = concept[:-1]
                            patterns.append(self.nlp(singular))
                            self.concept_to_ontology[singular] = onto_name
                        else:
                            plural = concept + 's'
                            patterns.append(self.nlp(plural))
                            self.concept_to_ontology[plural] = onto_name
            self.combined_matcher.add(onto_name, patterns)
            if self.debug:
                print(f"Added {len(patterns)} patterns for ontology: {onto_name}")

    def recognize_concepts(self, text):
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

    def get_concept_lineage(self, concept, onto_name):
        if concept in self.lineage_cache[onto_name]:
            return self.lineage_cache[onto_name][concept]

        onto = self.ontologies[onto_name]
        cls = onto.search_one(label=concept)

        if not cls:
            if self.debug:
                print(f"Could not find class for concept: {concept} in ontology: {onto_name}")
            return None

        superclasses = [c.label.first() for c in cls.ancestors() if c != cls]
        subclasses = [c.label.first() for c in cls.subclasses()]

        lineage = {
            "parents": superclasses,
            "children": subclasses
        }
        self.lineage_cache[onto_name][concept] = lineage
        return lineage

    def print_ontology_concepts(self, onto_name):
        if onto_name not in self.ontologies:
            print(f"Ontology {onto_name} not found.")
            return
        onto = self.ontologies[onto_name]
        print(f"Concepts in {onto_name}:")
        for cls in onto.classes():
            if cls.label:
                print(f"  - {cls.label}")

