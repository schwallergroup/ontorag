"""Convert OntoGen output to OWL format."""

from tree import Tree
import pickle
import owlready2
from owlready2 import Thing, types
from pydantic import BaseModel
import re


class OntoGen2Owl(BaseModel):
    def __call__(self, relationships, output_file='output_ontology.owl'):

        onto = self.create_ontology_from_relationships(relationships)

        print("\nClasses in the ontology before saving:")
        for cls in onto.classes():
            print(f"- {cls}")
            if cls.is_a:
                print(f"  Subclass of: {', '.join(str(parent) for parent in cls.is_a if parent != Thing)}")

        onto.save(file=output_file, format="rdfxml")
        print(f"\nOntology saved to '{output_file}'")

        # Attempt to reload the ontology
        print("\nAttempting to reload the ontology...")
        loaded_onto = owlready2.get_ontology(f"file://{output_file}").load()

        print("\nClasses in the reloaded ontology:")
        for cls in loaded_onto.classes():
            print(f"- {cls}")
            if cls.is_a:
                print(f"  Subclass of: {', '.join(str(parent) for parent in cls.is_a if parent != Thing)}")

        print(f"\nNumber of classes in reloaded ontology: {len(list(loaded_onto.classes()))}")

        return onto, loaded_onto

    def clean_class_name(self, name):
        words = re.findall(r'\w+', name)
        return ''.join(word.capitalize() for word in words)

    def create_ontology_from_relationships(self, relationships):
        onto = owlready2.get_ontology("http://example.org/example.owl")

        with onto:
            for relationship in str(relationships).split('\n'):
                try:
                    subclass, superclass = map(str.strip, relationship.split('isA'))
                    subclass_name = self.clean_class_name(subclass)
                    superclass_name = self.clean_class_name(superclass)

                    # Create or get the superclass
                    if superclass_name != "Thing":
                        super_cls = types.new_class(superclass_name, (Thing,))
                    else:
                        super_cls = Thing

                    # Create or get the subclass
                    sub_cls = types.new_class(subclass_name, (super_cls,))
                except:
                    continue

        return onto


if __name__ == "__main__":
    transf = OntoGen2Owl()
    fpath = 'data/ontologies/sacs_llama3.1:70b/'
    wmap_file = "wordmap_0.pkl"

    with open(fpath + wmap_file, "rb") as f: 
        tree = pickle.load(f)["Thing"] 

    transf(tree, output_file = fpath + "output_ontology_0.owl")