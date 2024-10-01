import pickle
import re
import owlready2
from owlready2 import Thing, types
from pydantic import BaseModel
import pandas as pd
from typing import Optional, Dict

class OntoGen2Owl(BaseModel):
    def __call__(self, wmap: str, definitions: Optional[str]=None, output_file="output_ontology.owl"):
        with open(wmap, "rb") as f:
            relationships = pickle.load(f)["Thing"]
        
        defs = {}
        if definitions:
            defs = pd.read_csv(definitions, names=["concept", "definition"]).set_index('concept')['definition'].to_dict()

        onto = self.create_ontology_from_relationships(relationships, defs)

        print("\nClasses in the ontology before saving:")
        for cls in onto.classes():
            print(f"- {cls}")
            if cls.is_a:
                print(f"  Subclass of: {', '.join(str(parent) for parent in cls.is_a if parent != Thing)}")
            if cls.IAO_0000115:
                print(f"  Definition: {cls.IAO_0000115[0]}")

        onto.save(file=output_file, format="rdfxml")
        print(f"\nOntology saved to '{output_file}'")

        print("\nAttempting to reload the ontology...")
        loaded_onto = owlready2.get_ontology(f"file://{output_file}").load()

        print("\nClasses in the reloaded ontology:")
        for cls in loaded_onto.classes():
            print(f"- {cls}")
            if cls.is_a:
                print(f"  Subclass of: {', '.join(str(parent) for parent in cls.is_a if parent != Thing)}")
            if cls.IAO_0000115:
                print(f"  Definition: {cls.IAO_0000115[0]}")

        print(f"\nNumber of classes in reloaded ontology: {len(list(loaded_onto.classes()))}")

        return onto, loaded_onto

    def clean_class_name(self, name):
        """Clean the class name."""
        words = re.findall(r"\w+", name)
        return "".join(word.capitalize() for word in words)

    def create_ontology_from_relationships(self, relationships, definitions: Optional[Dict[str, str]]=None):
        """Create an ontology from the relationships."""
        onto = owlready2.get_ontology("http://example.org/example.owl")

        with onto:
            class IAO_0000115(owlready2.AnnotationProperty):
                namespace = onto

            for relationship in str(relationships).split("\n"):
                try:
                    subclass, superclass = map(str.strip, relationship.split("isA"))
                    subclass_name = self.clean_class_name(subclass)
                    superclass_name = self.clean_class_name(superclass)

                    if superclass_name != "Thing":
                        super_cls = types.new_class(superclass_name, (Thing,))
                    else:
                        super_cls = Thing

                    sub_cls = types.new_class(subclass_name, (super_cls,))

                    if definitions and subclass_name in definitions:
                        sub_cls.IAO_0000115.append(definitions[subclass_name])
                        print(f"Added definition for {subclass_name}: {definitions[subclass_name]}")
                except Exception as e:
                    print(f"Error processing relationship: {relationship}. Error: {str(e)}")
                    continue

        return onto

if __name__ == "__main__":
    transf = OntoGen2Owl()
    for model in ['sacs_claude3_5', 'sacs_llama3.1:70b']:
        fpath = f"data/ontologies/{model}/" # path with the tree files 
        for i in range(5):
            wmap_file = f"wordmap_{i}.pkl" #wordmap to use 
            defs_file = "definitions.csv" # add definitions from csv 

            transf(
                wmap=fpath + wmap_file,
                definitions=fpath + defs_file,
                output_file=fpath + f"sacs_ontology_{i}.owl" # output 
            )