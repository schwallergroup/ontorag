import argparse
import logging
import random
from pathlib import Path

import ollama
from utils import read_text, write_text

logging.basicConfig(level=logging.INFO)


prompt_generate_cats = """
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
"""

prompt_correct_format_cats = """
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
"""

prompt_synthesize_cats = """
Given all these trees of categories, build a combined tree that captures the recurring categories and topics from the provided data, ensuring that only the most frequently mentioned categories are included.
In other words, create a more organized and concise list.
Do not include conjunctions in category names, e.g., "<Category 1> and <Category 2>" should be two separate categories: "<Category 1>" and "<Category 2>". 
Here is the list, separated by '-':

{CATEGORIES}

Remember your goal: build a combined tree that captures the recurring categories and topics from the provided data, ensuring that only the most frequently mentioned categories are included.
In other words, create a more organized and concise list.
Do not include conjunctions in category names, e.g., "<Category 1> and <Category 2>" should be two separate categories: "<Category 1>" and "<Category 2>". 
"""


prompt_self_reflect_fix_format_cats = """
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
"""


def llm_format_cats(
    cats,
    model,
    options,
):
    """
    This function receives a string with categories and subcategories which might be incorrectly formatted.
    The function uses an LLM to format the list of categories and subcategories to the same format,
    which is the following:

    Main Category 1: <Category name>

            Subcategory 1: <Subcategory name>
            Subcategory 2: <Subcategory name>

    Main Category 2: <Category name>

            Subcategory 1: <Subcategory name>


    The function will keep asking the user to fix the format until the format is correct or the maximum number of retries is reached.
    If the output of the LLM is incorrectly formated, a self reflection prompt is used to ask the model again to fix the format.

    Parameters:
    cats (str): String with categories and subcategories.
    model (str): Ollama model tag to use.
    options (dict): Dictionary with options to use in the model.
    """
    formated_prompt = prompt_correct_format_cats.format(CATEGORIES=cats)
    response = ollama.chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": formated_prompt,
            },
        ],
        options=options,
    )
    result = response["message"]["content"]

    max_retry = 3
    while "Main Category" not in result or "    Subcategory" not in result:
        response = ollama.chat(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": formated_prompt,
                },
                {
                    "role": "assistant",
                    "content": result,
                },
                {
                    "role": "user",
                    "content": prompt_self_reflect_fix_format_cats,
                },
            ],
            options=options,
        )
        result = response["message"]["content"]
        max_retry -= 1
        if max_retry == 0:
            break
    return result


def llm_synthesize_cats(cats, model, options):
    """
    This function receives a string with categories and subcategories and asks an LLM to
    synthesize the categories and subcategories into a more concise list with the most frequently mentioned categories.
    If the output of the LLM is incorrectly formated, the model is prompted again with the same question. This is
    done to avoid answers where the output is empty or non-sensical.

    Parameters:
    cats (str): String with categories and subcategories.
    model (str): Ollama model tag to use.
    options (dict): Dictionary with options to use in the model.
    """
    formated_prompt = prompt_synthesize_cats.format(CATEGORIES=cats)
    response = ollama.chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": formated_prompt,
            },
        ],
        options=options,
    )
    result = response["message"]["content"]
    if "Main Category" not in result or "    Subcategory" not in result:
        response = ollama.chat(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": formated_prompt,
                },
            ],
            options=options,
        )
        result = response["message"]["content"]
    return result


def parse_cats_and_subcats(text):
    """
    This function receives a string with categories and subcategories with the following format:

    Main Category 1: <Category name>

            Subcategory 1: <Subcategory name>
            Subcategory 2: <Subcategory name>

    Main Category 2: <Category name>

            Subcategory 1: <Subcategory name>

    and returns a dictionary with the categories and subcategories.

    Parameters:
    text (str): String with categories and subcategories.
    """
    cats = {}
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip().startswith("*"):
            line = line.replace("*", "")
        if line.startswith("Main Category"):
            cat = line.split(":")[-1].strip()
            cats[cat] = []
            i += 1
            while i < len(lines) and (
                lines[i].strip().startswith("Subcategory")
                or len(lines[i].strip()) == 0
            ):
                if len(lines[i].strip()) > 0:
                    subcat = lines[i].split(":")[-1].strip()
                    cats[cat].append(subcat)
                i += 1
        i += 1
    return cats


def cats_to_str(cats):
    """
    This function receives a dictionary with categories and subcategories and returns a string with the categories and subcategories,
    formatted as follows:

    <Main Category 1>
        <Subcategory 1>
        <Subcategory 2>
    --------------------
    <Main Category 2>
        <Subcategory 1>
        <Subcategory 2>
        <Subcategory 3>
    --------------------

    Parameters:
    cats (dict): Dictionary with categories and subcategories.
    """
    context = ""
    for answer in cats:
        if len(answer) == 0:
            continue
        for k in answer:
            k_clean = k.replace("*", "")
            context += k_clean + "\n"
            for sub in answer[k]:
                sub_clean = sub.replace("*", "")
                context += f"    {sub_clean}\n"
        context += "-" * 80 + "\n"
    return context


def generate_categories(
    txt_files,
    main_topic,
    generation_model,
    format_model,
    synthesis_model,
    num_retries_consistency=20,
    num_generated_seed=20,
    generation_model_options={},
    format_model_options={},
    synthesis_model_options={},
):
    """
    Generate multiple possible categories seeds from a list of txt files.

    Parameters:
    txt_files (list): List of txt files to process.
    main_topic (str): Main topic to generate categories for.
    generation_model (str): Ollama model tag to use to generate categories.
    format_model (str): Ollama model tag to use to format categories.
    synthesis_model (str): Ollama model tag to use to synthesize frequent categories.
    num_retries_consistency (int): Number of retries to use with self-consistency.
            A larger number will increase the chances of getting consistent results. (default: 20)
    num_generated_seed (int): Number of seeds to generate. (default: 20)
    generation_model_options (dict): Dictionary with options to use in the generation model.
    format_model_options (dict): Dictionary with options to use in the format model.
    synthesis_model_options (dict): Dictionary with options to use in the synthesis model.
    """

    # random shuffle the list of files
    random.shuffle(txt_files)

    context = ""
    for i, txt_file in enumerate(txt_files):
        text = read_text(txt_file)
        while "\n\n" in text:
            text = text.replace("\n\n", "\n")
        context += f"Introduction {i}: {text}\n\n"

    responses = []
    for retry in range(num_retries_consistency):
        logging.info(
            f"Generating categories Retry {retry + 1}/{num_retries_consistency}"
        )
        formated_prompt = prompt_generate_cats.format(
            CONTEXT=context, TOPIC=main_topic
        )
        response = ollama.chat(
            model=generation_model,
            messages=[
                {
                    "role": "user",
                    "content": formated_prompt,
                },
            ],
            options=generation_model_options,
        )
        responses.append(response["message"]["content"])
        logging.info(response["message"]["content"])

    formated_responses = []
    responses = [c for c in responses if len(c.strip()) > 0]
    for i, response in enumerate(responses):
        logging.info(f"Formatting category {i + 1}/{len(responses)}")
        response = llm_format_cats(
            response, format_model, options=format_model_options
        )
        formated_responses.append(response)
        logging.info(response)

    formated_responses_reduced = []
    for response in formated_responses:
        formated_responses_reduced.append(parse_cats_and_subcats(response))
    filtered_cats_text = cats_to_str(formated_responses_reduced)

    synthesized_cats = []
    for i in range(num_generated_seed):
        logging.info(f"Synthesizing categories {i + 1}/{num_generated_seed}")
        res = llm_synthesize_cats(
            filtered_cats_text,
            synthesis_model,
            options=synthesis_model_options,
        )
        synthesized_cats.append(res)
        logging.info(res)

    formated_responses = []
    responses = [c for c in responses if len(c.strip()) > 0]
    for i, response in enumerate(synthesized_cats):
        logging.info(f"Formatting category {i + 1}/{len(responses)}")
        response = llm_format_cats(
            response, format_model, options=format_model_options
        )
        formated_responses.append(response)
        logging.info(response)

    categories_folder = Path("categories")
    categories_folder.mkdir(exist_ok=True)
    for i, res in enumerate(formated_responses):
        write_text(
            categories_folder / f"{main_topic}_categories_seed.{i}.txt", res
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate possible categories seeds from a list of txt files."
    )
    parser.add_argument(
        "main_topic", type=str, help="Main topic to generate categories for."
    )
    parser.add_argument(
        "txt_files",
        type=str,
        nargs="+",
        help="Path to the text files to process.",
    )
    parser.add_argument(
        "--generation-model",
        "-gm",
        help="Ollama model tag to use to generate categories.",
        type=str,
    )
    parser.add_argument(
        "--generation-temperature",
        "-gt",
        help="Model temperature to use to generate categories.",
        type=float,
    )
    parser.add_argument(
        "--generation-num-ctx",
        "-gc",
        help="Context length in tokens to use to generate categories.",
        type=int,
    )
    parser.add_argument(
        "--format-model",
        "-fm",
        help="Ollama model tag to use to format categories.",
        type=str,
    )
    parser.add_argument(
        "--format-temperature",
        "-ft",
        help="Model temperature to use to format categories.",
        type=float,
    )
    parser.add_argument(
        "--format-num-ctx",
        "-fc",
        help="Context length in tokens to use to format categories.",
        type=int,
    )
    parser.add_argument(
        "--synthesis-model",
        "-sm",
        help="Ollama model tag to use to synthesize categories.",
        type=str,
    )
    parser.add_argument(
        "--synthesis-temperature",
        "-st",
        help="Model temperature to use to synthesize categories.",
        type=float,
    )
    parser.add_argument(
        "--synthesis-num-ctx",
        "-sc",
        help="Context length in tokens to use to synthesize categories.",
        type=int,
    )
    parser.add_argument(
        "--num-retries",
        "-r",
        help="Number of retries to ensure consistency.",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--num-generated-seed",
        "-s",
        help="Number of generated seeds.",
        type=int,
        default=20,
    )

    args = parser.parse_args()
    main_topic = args.main_topic
    txt_files = args.txt_files
    num_retries = args.num_retries
    num_generated_seed = args.num_generated_seed

    generation_model = args.generation_model
    format_model = args.format_model
    synthesis_model = args.synthesis_model

    options_generation = {}
    if args.generation_temperature:
        options_generation["temperature"] = args.generation_temperature
    if args.generation_num_ctx:
        options_generation["num_ctx"] = args.generation_num_ctx

    options_format = {}
    if args.format_temperature:
        options_format["temperature"] = args.format_temperature
    if args.format_num_ctx:
        options_format["num_ctx"] = args.format_num_ctx

    options_synthesis = {}
    if args.synthesis_temperature:
        options_synthesis["temperature"] = args.synthesis_temperature
    if args.synthesis_num_ctx:
        options_synthesis["num_ctx"] = args.synthesis_num_ctx

    generate_categories(
        txt_files,
        main_topic,
        generation_model,
        format_model,
        synthesis_model,
        num_retries_consistency=num_retries,
        num_generated_seed=num_generated_seed,
        generation_model_options=options_generation,
        format_model_options=options_format,
        synthesis_model_options=options_synthesis,
    )
