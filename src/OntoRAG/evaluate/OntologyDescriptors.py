from owlready2 import *
from collections import defaultdict

def analyze_ontology(ontology_path):
    # Load the ontology
    onto = oget_ontology(ontology_path).load()

    # Get classes, individuals, and properties
    classes = list(onto.classes())
    individuals = list(onto.individuals())
    properties = list(onto.properties())

    # Calculate depth and children statistics
    class_depths = {}
    class_children = defaultdict(list)
    max_depth = 0
    
    def calculate_depth(cls):
        if cls in class_depths:
            return class_depths[cls]
        
        if len(cls.is_a) == 0:  # Top-level class
            depth = 0
        else:
            depth = 1 + max(calculate_depth(parent) for parent in cls.is_a if isinstance(parent, ThingClass))
        
        class_depths[cls] = depth
        return depth

    for cls in classes:
        depth = calculate_depth(cls)
        max_depth = max(max_depth, depth)
        
        for child in cls.subclasses():
            class_children[cls].append(child)

    # Calculate children statistics
    max_children = max(len(children) for children in class_children.values())
    total_children = sum(len(children) for children in class_children.values())
    avg_children = total_children / len(classes) if classes else 0

    single_child_classes = [cls for cls, children in class_children.items() if len(children) == 1]
    above_avg_children_classes = [cls for cls, children in class_children.items() if len(children) > avg_children]

    return {
        "classes": classes,
        "individuals": individuals,
        "properties": properties,
        "max_depth": max_depth,
        "max_children": max_children,
        "avg_children": avg_children,
        "single_child_classes": single_child_classes,
        "above_avg_children_classes": above_avg_children_classes
    }

def main():
    ontology_path = "/home/matt/Proj/Hermetica/Ontology/Ontologies/PolymerOntology.owl"
    results = analyze_ontology(ontology_path)

    print(f"Number of Classes: {len(results['classes'])}")
    print(f"Number of Individuals: {len(results['individuals'])}")
    print(f"Number of Properties: {len(results['properties'])}")
    print(f"Maximum Depth: {results['max_depth']}")
    print(f"Maximum Number of Children: {results['max_children']}")
    print(f"Average Number of Children: {results['avg_children']:.2f}")
    print(f"Classes with a Single Child: {len(results['single_child_classes'])}")
    print(f"Classes with More than Average Children: {len(results['above_avg_children_classes'])}")

if __name__ == "__main__":
    main()