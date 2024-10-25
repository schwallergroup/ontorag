import argparse
import sys
import logging
from pathlib import Path
from utils import read_text, write_text, write_tuples_list_to_csv
sys.path.insert(0, sys.path[0] + '/termo')
from termo import Termo

logging.basicConfig(level=logging.INFO)


def extract_terms(
        text_file, 
        model=None, 
        max_length_split_terms=2000,
        max_length_split_acronyms=2000,
        max_length_split_definitions=2000,
        max_length_split_relationships=2000,
        model_params={},
        skip_acronyms=False,
        skip_definitions=False,
        skip_relationships=False):
    text_file = Path(text_file)
    term_file = text_file.parent / (text_file.stem + '.terms.csv')
    defi_file = text_file.parent / (text_file.stem + '.definitions.csv')
    acro_file = text_file.parent / (text_file.stem + '.acronyms.csv')
    rela_file = text_file.parent / (text_file.stem + '.relationships.csv')

    logging.info(f'Extracting terms from {text_file} using model {model}')
    text = read_text(text_file)
    termo = Termo(text)
    terms = termo.extract_terms(model=model, max_length_split=max_length_split_terms, options=model_params)
    write_tuples_list_to_csv(term_file, terms)

    if not skip_acronyms:
        logging.info(f'Extracting acronyms from {text_file} using model {model}')
        acronyms = termo.extract_acronyms(model=model, max_length_split=max_length_split_acronyms, options=model_params)
        write_tuples_list_to_csv(acro_file, list(acronyms.items()))

    if not skip_definitions:
        logging.info(f'Extracting definitions from {text_file} using model {model}')
        definitions = termo.extract_definitions(model=model, max_length_split=max_length_split_definitions, options=model_params)
        write_tuples_list_to_csv(defi_file, list(definitions.items()))

    if not skip_relationships:
        logging.info(f'Extracting relationships from {text_file} using model {model}')
        relationships = termo.extract_relationships(model=model, max_length_split=max_length_split_relationships, options=model_params)
        write_tuples_list_to_csv(rela_file, relationships)        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract vocabulary, definitions, acronyms and relationships from a txt file.')
    parser.add_argument('model', type=str, help='Ollama model tag to use.')
    parser.add_argument('txt_file', type=str, help='Path to the text file to process.')
    parser.add_argument('--max_length_split_terms', help='Maximum length (in characters) of the text to split for terms extraction.', type=int, default=2000)
    parser.add_argument('--max_length_split_acronyms', help='Maximum length (in characters) of the text to split for acronyms extraction.', type=int, default=2000)
    parser.add_argument('--max_length_split_definitions', help78='Maximum length (in characters) of the text to split for definitions extraction.', type=int, default=2000)
    parser.add_argument('--max_length_split_relationships', help='Maximum length (in characters) of the text to split for relationships extraction.', type=int, default=2000)
    parser.add_argument('--temperature', '-t', help='Model temperature to use.', type=float)
    parser.add_argument('--num_ctx', '-n', help='Context length in tokens to use.', type=int)
    parser.add_argument('--skip-acronyms', '-a', help='Skip acronyms extraction.', action='store_true')
    parser.add_argument('--skip-definitions', '-d', help='Skip definitions extraction.', action='store_true')
    parser.add_argument('--skip-relationships', '-r', help='Skip relationships extraction.', action='store_true')

    args = parser.parse_args()
    txt_files = args.txt_file
    model = args.model
    model_params = {}
    if args.temperature:
        model_params['temperature'] = args.temperature
    if args.num_ctx:
        model_params['num_ctx'] = args.num_ctx
    
    extract_terms(
        args.txt_file, 
        model, 
        args.max_length_split_terms,
        args.max_length_split_acronyms,
        args.max_length_split_definitions,
        args.max_length_split_relationships,
        model_params,
        skip_acronyms=args.skip_acronyms,
        skip_definitions=args.skip_definitions,
        skip_relationships=args.skip_relationships
    )
