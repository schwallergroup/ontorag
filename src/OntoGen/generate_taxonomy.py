#import anthropic
import os
import requests
import argparse
import re
import ollama
import random
import logging
from collections import defaultdict
from unidecode import unidecode
from collections import defaultdict
from utils import read_tuples_list_from_csv
from pathlib import Path
from tree import Tree, lemma


logging.basicConfig(level=logging.INFO)


prompt = '''
Given this context:

===
{CONTEXT}
===

,And given the following taxonomy:

===
{TAXONOMY}
===

Complete the following list to classify the terms into the taxonomy according to the context.
If a term does not fit in any of the categories, say "None".
If the text is not clear enough to classify a term, say "None".
If the text does not explicitly mention that a term is a type of another, say "None".
If multiple categories apply, choose the most specific one.
In the ouput include only the classification. Do not include any explanation or additional information.
Do not classify a term to be its own parent, this is, do not output answers such as "A isA A".
The answer should include 'isA'.


{TERMS}

'''


def get_initial_categories(category_seed_file):
    '''
    Extract list of categories from the category seed file formated as follows:
    
    Main Category: Category 1
        ...
    Main Category: Category N
        ...

    Parameters:
    category_seed_file (str): Path to the category seed file.
    '''
    cats = []
    with open(category_seed_file, 'r') as f:
        lines = f.readlines()
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.strip().startswith('*'):
                line = line.replace('*', '')
            if line.startswith('Main Category'):
                cat = line.split(':')[-1].strip()
                cats.append(cat)
            i += 1
    return cats


def universal_id(word):
    '''
    Get unique id for a word. The id is obtained by lowercasing the word, removing dashes, and lemmatizing it.

    Parameters:
    word (str): Word to get the unique id.
    '''
    word = word.replace('-', ' ')
    words = word.split(' ')
    for i in range(len(words)):
        if len(words[i]) > 1:
            w = words[i]
            if w[0].isupper() and w[1].islower(): # if the first letter is uppercase and the rest is lowercase -> set all to lowercase
                words[i] = w.lower()
    word = ' '.join(words)
    return lemma(word)


def process_vocabulary(vocabulary, acronyms):
    '''
    Process the vocabulary by removing acronyms, dashes, and asterisks, while keeping all the transformations applied as synonyms.
    Return the processed vocabulary and a dictionary with the id to synonyms.

    Parameters:
    vocabulary (list): List of terms to process.
    acronyms (dict): Dictionary with acronyms and their corresponding terms.

    Returns:
    list: Processed vocabulary.
    dict: Dictionary with the id to synonyms.
    '''
    id_to_synonyms = defaultdict(set)

    # remove asterisks as these are probably artifacts from vocabulary extraction
    vocabulary = [v.replace('*', '') for v in vocabulary]
    # remove acronyms from the vocabulary
    for acr in acronyms:
        if acr in vocabulary:
            vocabulary.remove(acr)

    for acr, term in acronyms.items():
        id_to_synonyms[universal_id(term)].add(acr)
        id_to_synonyms[universal_id(term)].add(term)

    vocabulary = [(v, v) for v in vocabulary] # old and new vocabulary
    # clean vocabulary from acronyms
    for i in range(len(vocabulary)):
        for acronym, term in acronyms.items():
            oldv, newv = vocabulary[i]
            if f"({acronym})" in oldv:
                vocabulary[i] = oldv, newv.replace(f"({acronym})", '').strip()

    for oldv, newv in vocabulary:
        id_to_synonyms[universal_id(newv)].add(oldv)
        id_to_synonyms[universal_id(newv)].add(newv)

    # from now we use the no-acronym version as old vocabulary
    vocabulary = [(v2, v2) for v1, v2 in vocabulary]

    # remove dashes
    vocabulary = [(v1, v2.replace('-', ' ')) for v1, v2 in vocabulary]

    for oldv, newv in vocabulary:
        id_to_synonyms[universal_id(oldv)].add(newv)

    lower_old_vocabulary = [v1.lower() for v1, v2 in vocabulary]
    filtered_vocabulary = []
    already_present = set()
    for v1, v2 in vocabulary:
        if v1.lower() not in already_present:
            filtered_vocabulary.append((v1, v2))
            already_present.add(v1.lower())
    vocabulary = filtered_vocabulary
    
    return [v1 for v1, v2 in vocabulary], id_to_synonyms


def query(model, context, taxonomy, terms, options={}):
    '''
    Query the taxonomic relations of a list of terms given a context and a taxonomy,
    using an Ollama model.

    Parameters:
    model (str): Ollama model tag to use.
    context (str): Context to use in the query.
    taxonomy (str): Taxonomy to use in the query.
    terms (list): List of terms to classify.
    options (dict): Options to use in the query.

    Returns:
    str: Response from the Ollama model.
    '''
    formated_prompt = prompt.format(CONTEXT=context, TAXONOMY=taxonomy, TERMS='\n'.join([t + ' isA ' for t in terms]))
    logging.info("PROMPT:" + formated_prompt)
    response = ollama.chat(model=model, messages=[
        {
            'role': 'user',
            'content': formated_prompt,
        },
    ], options=options)
    return response['message']['content']


def parse_answer(answer):
    '''
    Parse the answer from the Ollama model to extract the taxonomic relations.
    Receives the answer and returns a dictionary with the terms and their corresponding categories.
    The expected input has relationship in the form of 'A isA B' or 'A is B' or 'A: B'.

    Parameters:
    answer (str): Answer from the Ollama model.

    Returns:
    dict: Dictionary with the terms and their corresponding categories.
    '''
    lines = answer.split('\n')
    res = {}
    for l in lines:
        print(l)
        if 'isA' in l or ':' in l or 'is' in l:
            separator = None
            if ':' in l:
                separator = ':'
            elif 'isA' in l:
                separator = 'isA'
            elif 'is' in l:
                separator = 'is'
            parts = l.split(separator)
            cat = parts[1].replace('*', '').strip()
            query = parts[0].replace('*', '').strip()
            pattern = re.compile(r'\d+\.\s')
            if pattern.match(query):
                query = query.split('. ')[1].strip()
            if len(cat) == 0 or len(query) == 0:
                continue
            if 'None' in cat:
                res[query] = 'None'
            else:
                res[query] = cat
    return res


def add_asnwers(wordmap, answers, root, destructive=False):
    '''
    Add taxonomic relationships to the wordmap dictionary of terms and tree nodes.
    A relationship is discarded if it creates a loop in the tree or if the parent is already present in the child
    or if both terms are the same.

    Parameters:
    wordmap (dict): Dictionary of terms and tree nodes.
    answers (dict): Dictionary of terms and their corresponding categories. 
    root (Tree): Root of the tree.
    destructive (bool): If True, deletes a node from the tree if it was already present in the tree and creates a new one for each relationship.

    Returns:
    dict: Updated wordmap dictionary.
    '''
    for k, v in answers.items():
        if 'None' in v:
            continue
        if k not in wordmap or v not in wordmap:
            continue
        if wordmap[v] == wordmap[k]:
            continue
        # sanity check: make sure that no loops are created
        childs = wordmap[k].list_all_childs()
        if wordmap[v] in childs:
            continue
        # sanity check: direct parent is not present
        if wordmap[v] in wordmap[k].children:
            continue
        # sanity check: child is already present
        if wordmap[k] in wordmap[v].children:
            continue
        if not destructive:
            wordmap[v].add_child(wordmap[k])
        else:
            if root.exists(k):
                root.remove_node(k)
            wordmap[v].add_child(wordmap[k])
    return wordmap
            

def remove_self_loops_tree(tree):
    '''
    Recursive function to remove self loops from a tree.

    Parameters:
    tree (Tree): Tree to remove self loops.
    '''
    for c in tree.children:
        c.children = [cc for cc in c.children if cc != tree]
        c = remove_self_loops_tree(c)
    return tree


def majority_voting_answers(wordmap, answers_list):
    '''
    Perform majority voting on a list of answers to get the most common taxonomic relationships.
    The majority is calculated as the half of the number of answers plus one.
    Relationships are discarded if they are not present in the wordmap or if the terms are the same.
    
    Parameters:
    wordmap (dict): Dictionary of terms and tree nodes.
    answers_list (list): List of dictionaries with terms and their corresponding categories.

    Returns:
    dict: Dictionary with the most common taxonomic relationships.
    '''
    res = {}
    majority = len(answers_list) // 2 + 1

    count = defaultdict(int)
    for answers in answers_list:
        for k, v in answers.items():
            if k == 'None' or v == 'None':
                continue
            if k.lower() != v.lower() and k in wordmap and v in wordmap and wordmap[k] != wordmap[v]:
                count[(k, v)] += 1

    for k,v in count.items():
        if v >= majority:
            res[k[0]] = k[1]

    return res


def generate_taxonomy(
        category_seed_file, 
        txt_files, 
        model, 
        model_params,
        num_iterations,
        prompt_include_path):
    '''
    Generate a taxonomy from a list of text files using an Ollama model.

    Parameters:
    category_seed_file (str): Path to the category seed txt file.
    txt_files (list): List of paths to the text files to process.
    model (str): Ollama model tag to use to generate categories.
    model_params (dict): Additional parameters to pass to the Ollama model.
    num_iterations (int): Number of iterations to run for each text file. The majority answer from the answers from the iterations is used.
    prompt_include_path (bool): Include a term path in the taxonomy (i.e. parent categories). 
            If True, the parent categories are included in the taxonomy. This provides more context for the model but 
            might bias the answers towards the parent categories, while making the output taxonomy more consistent and
            less prone to errors.
    '''

    cats = get_initial_categories(category_seed_file)

    tree = Tree('Thing')
    for t in cats:
        tree.children.append(Tree(t))

    title_to_terms = defaultdict(set)
    title_to_acron = defaultdict(dict)
    title_to_context = {}

    for txt in txt_files:
        txt_path = Path(txt)
        title = txt_path.stem
        
        root_folder = txt_path.parent
        terms_csv = os.path.join(root_folder, f'{title}.terms.csv')
        acron_csv = os.path.join(root_folder, f'{title}.acronyms.csv')

        terms = read_tuples_list_from_csv(terms_csv)
        acron = read_tuples_list_from_csv(acron_csv)
        title_to_terms[txt] = set([t[0] for t in terms])
        title_to_acron[txt] = {t[0]: t[1] for t in acron}
        title_to_context[txt] = txt_path.read_text()

    merged_ids_to_synonyms = defaultdict(set)
    for title in title_to_terms.keys():
        vocabulary = list(title_to_terms[title])
        acronyms = title_to_acron[title]
        vocabulary, id_to_synonyms = process_vocabulary(vocabulary, acronyms)
        # merge 2 dictionaries
        for k, v in id_to_synonyms.items():
            merged_ids_to_synonyms[k] = merged_ids_to_synonyms[k].union(v)

    wordmap = {}
    repeated_ids = set()
    for k, v in merged_ids_to_synonyms.items():
        t = Tree(list(v)[0])
        t.synonyms = list(v)
        for syn in v:
            if syn in wordmap:
                repeated_ids.add(syn)
                ## remove the synonym from the wordmap
                #t = wordmap.pop(syn) 
                # make wordmap point to the previous tree (the current one will be ignored)
                print("Merging nodes: ", wordmap[syn].synonyms, t.synonyms)
                wordmap[syn] = t

            else:
                wordmap[syn] = t

    tree_terms = tree.get_terms()
    # remove 'Thing' from the tree list
    tree_terms.remove('Thing')

    # set wordmap entries for existing tree
    for node in tree.get_nodes_list():
        wordmap[node.synonyms[0]] = node

    list_titles = list(title_to_terms.keys())
    list_titles.sort()

    for iteration in range(num_iterations):
        logging.info('#' * 80)
        logging.info(f"    Level {iteration}")
        logging.info('#' * 80)

        # randomly shuffle list_titles
        random.shuffle(list_titles)

        for j in range(len(list_titles)):
            answers_list = []

            for retry in range(3):

                title = list_titles[j]
                context = title_to_context[title]

                tree_terms, tree_paths = tree.get_terms_and_paths(ctx=context)
                #tree_terms, tree_paths = tree.get_level_terms_and_path(level + 1, ctx=context)

                vocabulary, _ = process_vocabulary(list(title_to_terms[title]), title_to_acron[title])
                vocabulary = [v for v in vocabulary if v not in repeated_ids] 
                #vocabulary = [list(v)[0] for k,v in merged_ids_to_synonyms.items()] # whole vocabulary
                terms = vocabulary

                # remove 'Thing' from the tree list and tree path
                try:
                    index = tree_terms.index('Thing')
                    tree_terms.pop(index)
                    tree_paths.pop(index)
                except ValueError as e:
                    # if 'Thing' is not in the list
                    pass

                for i in range(len(tree_paths)):
                    # remove the self word from the path
                    tree_paths[i].remove(tree_terms[i])
                    # remove 'Thing' from each path
                    if 'Thing' in tree_paths[i]:
                        tree_paths[i].remove('Thing')

                # filter those terms that are already in the tree to avoid influencing the answer
                # we only filter those that are leafs, this is, we remove the "instances"
                filtered_tree_terms = []
                filtered_tree_paths = []
                tree_terms_nodes = []
                for t in tree_terms:
                    if t in wordmap:
                        tree_terms_nodes.append(wordmap[t])
                for i in range(len(tree_terms)):
                    if tree_terms[i] not in wordmap:
                        logging.info(f"Skip: Term {tree_terms[i]} not in wordmap")
                        continue
                    if wordmap[tree_terms[i]] not in tree_terms_nodes or len(wordmap[tree_terms[i]].children) > 0:
                        filtered_tree_terms.append(tree_terms[i])
                        filtered_tree_paths.append(tree_paths[i])
                    elif tree_terms[i] in cats:
                        filtered_tree_terms.append(tree_terms[i])
                        filtered_tree_paths.append(tree_paths[i])
                    else:
                        pass

                # comment the lines below to avoid filtering
                #tree_terms = filtered_tree_terms
                #tree_paths = filtered_tree_paths

                # add path to terms names
                for i in range(len(tree_terms)):
                    if len(tree_paths[i]) > 0:
                        # this gets too long, we use just the first (or last) level
                        #joined_tree_paths = ', '.join(tree_paths[i])
                        #joined_tree_paths = tree_paths[i][0] # first level
                        joined_tree_paths = tree_paths[i][-1] # last level
                        if prompt_include_path:
                            tree_terms[i] += f" ({joined_tree_paths})"

                #taxonomy = '\n'.join(tree_terms) + '\n' + '\n'.join(vocabulary)
                taxonomy = '\n'.join(tree_terms)   

                response = query(model, context, taxonomy, terms, model_params)
                logging.info("RESPONSE: " + response)
                
                answers = parse_answer(response)
                answers_list.append(answers)
                logging.info("\nANSWERS:" + '\n'.join([f"{k} isA {v}" for k,v in answers.items()]))

            answers = majority_voting_answers(wordmap, answers_list)
            logging.info("\nMAJORITY VOTING ANSWERS:" + '\n'.join([f"{k} isA {v}" for k,v in answers.items()]))
            wordmap = add_asnwers(wordmap, answers, tree, destructive=False)
            tree = remove_self_loops_tree(tree)

            # taxonomy directory
            taxonomy_dir = Path(f"taxonomy")
            taxonomy_dir.mkdir(parents=True, exist_ok=True)

            import pickle
            with open(f"taxonomy/wordmap_{iteration}.pkl", 'wb') as f:
                pickle.dump(wordmap, f)
            with open(f"taxonomy/tree_{iteration}.pkl", 'wb') as f:
                pickle.dump(tree, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('category_seed_file', type=str, help='Path to the category seed txt file.')
    parser.add_argument('txt_files', type=str, nargs='+', help='Path to the text files to process.')
    parser.add_argument('--model', '-m', help='Ollama model tag to use to generate categories.', type=str)
    parser.add_argument('--temperature', '-t', help='Model temperature to use to generate categories.', type=float)
    parser.add_argument('--num-ctx', '-c', help='Context length in tokens to use to generate categories.', type=int)
    parser.add_argument('--num-iterations', '-i', help='Number of iterations to run.', type=int, default=5)
    parser.add_argument('--prompt-include-path', '-p', help='Include a term path in the taxonomy (i.e. parent categories).', action='store_true')

    args = parser.parse_args()
    category_seed_file = args.category_seed_file
    txt_files = args.txt_files
    num_iterations = args.num_iterations
    model = args.model
    model_params = {}
    if args.temperature:
        model_params['temperature'] = args.temperature
    if args.num_ctx:
        model_params['num_ctx'] = args.num_ctx
    prompt_include_path = args.prompt_include_path

    generate_taxonomy(
        category_seed_file, 
        txt_files, 
        model, 
        model_params,
        num_iterations,
        prompt_include_path
    )
