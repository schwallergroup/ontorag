import argparse
import random
import ollama
from utils import read_text, write_text
from pathlib import Path


prompt_generate_cats = '''
I will provide you with texts from a collection of scientific papers of '{TOPIC}'. 
Your task is to analyze the text and identify the most relevant and general terms that define the key aspects of '{TOPIC}'. 
You should categorize these terms into main categories and their respective subcategories.

Please follow these steps:

    Read the abstracts and introductions carefully.
    Identify the most significant terms that are frequently mentioned and are central to '{TOPIC}'.
    Organize these terms into main categories, with each main category containing a few subcategories.
    Ensure that the categories and subcategories are simple and brief.

The output should be structured as follows:

Main Category 1: <Category name>

    Subcategory 1: <Subcategory name>
    Subcategory 2: <Subcategory name>

Main Category 2: <Category name>

    Subcategory 1: <Subcategory name>
    Subcategory 2: <Subcategory name>

Continue this pattern as needed to cover the primary aspects of '{TOPIC}'.
Do not conjunctions in category names, e.g., "<Category 1> and <Category 2>" should be two separate categories: "<Category 1>" and "<Category 2>".

Texts:
{CONTEXT}
'''

prompt_correct_format_cats = '''
Given the following categories, format the categories and subcategories following the syntax below:

Main Category 1: <Category name>

    Subcategory 1: <Subcategory name>
    Subcategory 2: <Subcategory name>

Main Category 2: <Category name>

    Subcategory 1: <Subcategory name>
    Subcategory 2: <Subcategory name>
    Subcategory 3: <Subcategory name>

Main Category 3: <Category name>

    Subcategory 1: <Subcategory name>

Continue this pattern as needed to cover all the categories.
If a subcategory has subcategories, do not include them in the list. This is, include only the first and second level of the tree.

Here are the categories and subcategories:
{CATEGORIES}
'''

prompt_synthesize_cats = '''
Given all these trees of categories, build a combined tree that captures the recurring categories and topics from the provided data, ensuring that only the most frequently mentioned categories are included.
In other words, create a more organized and concise list.
Do not include conjunctions in category names, e.g., "<Category 1> and <Category 2>" should be two separate categories: "<Category 1>" and "<Category 2>". 
Here is the list, separated by '-':

{CATEGORIES}

Remember your goal: build a combined tree that captures the recurring categories and topics from the provided data, ensuring that only the most frequently mentioned categories are included.
In other words, create a more organized and concise list.
Do not include conjunctions in category names, e.g., "<Category 1> and <Category 2>" should be two separate categories: "<Category 1>" and "<Category 2>". 
'''


prompt_self_reflect_fix_format_cats = '''
Can you stop any error in the previous formatting? Recall that the format should be as follows:
---
Main Category 1: <Category name goes here>

    Subcategory 1: <Subcategory name goes here>
    Subcategory 2: <Subcategory name goes here>

Main Category 2: <Category name goes here>

    Subcategory 1: <Subcategory name goes here>
---
This is, the word "Main Category" should be followed by a number and a colon, and then the name of the category. The word "Subcategory" should be followed by a number, a colon, and then the name of the subcategory.
If you find any mistakes, please rewrite the whole list following the correct format. If you don't find any mistakes, respond with "The previous formatting is correct."
'''


def llm_format_cats(
        cats,
        model,
        options,
        ):

    formated_prompt = prompt_correct_format_cats.format(CATEGORIES=cats)
    response = ollama.chat(model=model, messages=[
        {
            'role': 'user',
            'content': formated_prompt,
        },
    ], options=options)
    result = response['message']['content']

    max_retry = 3
    while 'Main Category' not in result or '    Subcategory' not in result:
        response = ollama.chat(model=model, messages=[
        {
            'role': 'user',
            'content': formated_prompt,
        },
        {
            'role': 'assistant',
            'content': result,
        },
        {
            'role': 'user',
            'content': prompt_self_reflect_fix_format_cats,
        }
        ], options=options)
        result = response['message']['content']
        max_retry -= 1
        if max_retry == 0:
            break
    return result


def llm_synthesize_cats(
        cats, 
        model,
        options
        ):

    formated_prompt = prompt_synthesize_cats.format(CATEGORIES=cats)
    response = ollama.chat(model=model, messages=[
        {
            'role': 'user',
            'content': formated_prompt,
        },
    ], options=options)
    result = response['message']['content']
    if 'Main Category' not in result or '    Subcategory' not in result:
        response = ollama.chat(model=model, messages=[
            {
                'role': 'user',
                'content': formated_prompt,
            },
        ], options=options)
        result = response['message']['content']
    return result


def parse_cats_and_subcats(text):
    cats = {}
    lines = text.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip().startswith('*'):
            line = line.replace('*', '')
        if line.startswith('Main Category'):
            cat = line.split(':')[-1].strip()
            cats[cat] = []
            i += 1
            while i < len(lines) and (lines[i].strip().startswith('Subcategory') or len(lines[i].strip()) == 0):
                if len(lines[i].strip()) > 0:
                    subcat = lines[i].split(':')[-1].strip()
                    cats[cat].append(subcat)
                i += 1
        i += 1
    return cats


def cats_to_str(cats):
    context = ''
    for answer in cats:
        if len(answer) == 0:
            continue
        for k in answer:
            k_clean = k.replace('*', '')
            context += k_clean + '\n'
            for sub in answer[k]:
                sub_clean = sub.replace('*', '')
                context += f"    {sub_clean}\n"
        context += '-' * 80 + '\n'
    return context


def generate_categories(
        txt_files, 
        main_topic, 
        generation_model, 
        format_model, 
        num_retries_consistency=20, 
        num_generated_seed=20,
        model=None, 
        temperature=None,
        generation_model_options={},
    ):

    # random shuffle the list of files
    random.shuffle(txt_files)

    context = ""
    for i, txt_file in enumerate(txt_files):
        text = read_text(txt_file)
        while '\n\n' in text:
            text = text.replace('\n\n', '\n')
        context += f"Introduction {i}: {text}\n\n"
    
    responses = []
    for retry in range(num_retries_consistency):
        formated_prompt = prompt_generate_cats.format(CONTEXT=context, TOPIC=main_topic)
        response = ollama.chat(model=generation_model, messages=[
            {
                'role': 'user',
                'content': formated_prompt,
            },
        ], options=generation_model_options)
        responses.append(response['message']['content'])

    formated_responses = []
    responses = [c for c in responses if len(c.strip()) > 0]
    for response in responses:
        response = llm_format_cats(response)
        formated_responses.append(response)

    formated_responses_reduced = []
    for response in formated_responses:
        formated_responses_reduced.append(parse_cats_and_subcats(response))
    filtered_cats_text = cats_to_str(formated_responses_reduced)
    
    results = []
    for i in range(num_generated_seed):
        res = llm_synthesize_cats(filtered_cats_text)
        results.append(res)
        
    categories_folder = Path('categories')
    categories_folder.mkdir(exist_ok=True)
    for i, res in enumerate(results):
        write_text(categories_folder / f'{main_topic}_categories_seed.{i}.txt', res)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate possible categories seeds from a list of txt files.')
    parser.add_argument('main_topic', type=str, help='Main topic to generate categories for.')
    parser.add_argument('txt_files', type=str, nargs='+', help='Path to the text files to process.')
    parser.add_argument('--model', '-m', help='Ollama model tag to use.', type=str)
    parser.add_argument('--temperature', '-t', help='Model temperature to use.', type=float)

    args = parser.parse_args()
    main_topic = args.main_topic
    txt_files = args.txt_files
    model = args.model

    options = {}
    if args.temperature:
        options['temperature'] = args.temperature

    generate_categories(
        txt_files, 
        main_topic, 
        model, 
        model, 
        model=model, 
        temperature=args.temperature,
        generation_model_options=options
    )
    

