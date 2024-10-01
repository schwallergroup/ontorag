"""Convert OntoGen output to OWL format."""

import pickle
import re

import owlready2
from owlready2 import Thing, types
from pydantic import BaseModel
from tree import Tree
from typing import Optional, Dict
import pandas as pd


class OntoGen2Owl(BaseModel):
    def __call__(self, wmap: str, definitions: Optional[str]=None, output_file="output_ontology.owl"):

        with open(wmap, "rb") as f:
            relationships = pickle.load(f)["Thing"]
        if definitions:
            defs = pd.read_csv(definitions, names=["concept", "definition"]).set_index('concept')['definition'].to_dict()

        onto = self.create_ontology_from_relationships(relationships, defs)

        print("\nClasses in the ontology before saving:")
        for cls in onto.classes():
            print(f"- {cls}")
            if cls.is_a:
                print(
                    f"  Subclass of: {', '.join(str(parent) for parent in cls.is_a if parent != Thing)}"
                )

        onto.save(file=output_file, format="rdfxml")
        print(f"\nOntology saved to '{output_file}'")

        # Attempt to reload the ontology
        print("\nAttempting to reload the ontology...")
        loaded_onto = owlready2.get_ontology(f"file://{output_file}").load()

        print("\nClasses in the reloaded ontology:")
        for cls in loaded_onto.classes():
            print(f"- {cls}")
            if cls.is_a:
                print(
                    f"  Subclass of: {', '.join(str(parent) for parent in cls.is_a if parent != Thing)}"
                )

        print(
            f"\nNumber of classes in reloaded ontology: {len(list(loaded_onto.classes()))}"
        )

        return onto, loaded_onto

    def add_definitions(self, onto, defs):
        """Add definitions to the ontology."""
        for cls in onto.classes():
            if cls.name in defs.keys():
                cls.comment = defs[cls.name]
                print(f"#################Found {cls.name}")
        return onto

    def clean_class_name(self, name):
        """Clean the class name."""
        words = re.findall(r"\w+", name)
        return "".join(word.capitalize() for word in words)

    def create_ontology_from_relationships(self, relationships, descriptions: Optional[Dict[str, str]]=None):
        """Create an ontology from the relationships."""
        onto = owlready2.get_ontology("http://example.org/example.owl")

        with onto:
            class IAO_0000115(owlready2.AnnotationProperty):
                namespace = onto

            for relationship in str(relationships).split("\n"):
                try:
                    subclass, superclass = map(
                        str.strip, relationship.split("isA")
                    )
                    subclass_name = self.clean_class_name(subclass)
                    superclass_name = self.clean_class_name(superclass)

                    # Create or get the superclass
                    if superclass_name != "Thing":
                        super_cls = types.new_class(superclass_name, (Thing,))
                    else:
                        super_cls = Thing

                    # Create or get the subclass
                    sub_cls = types.new_class(subclass_name, (super_cls,))

                    # Add description if available
                    if descriptions and subclass_name in descriptions:
                        sub_cls.IAO_0000115.append(descriptions[subclass_name])
                except:
                    continue

        return onto


if __name__ == "__main__":
    transf = OntoGen2Owl()
    fpath = "data/ontologies/sacs_claude3_5/"
    wmap_file = "wordmap_0.pkl"
    defs_file = "definitions_claude.csv"

    transf(
        wmap=fpath + wmap_file,
        definitions=fpath + defs_file,
        output_file=fpath + "output_ontology_0.owl"
    )
