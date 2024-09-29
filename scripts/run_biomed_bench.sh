#!/bin/bash

llms=(
    "mistral/mistral-medium-latest"
    # "anthropic/claude-3-5-sonnet-20240620"
    # "openai/gpt-3.5-turbo"
    # "openai/gpt-4-turbo-preview"
    "openai/gpt-4o"
    # "openai/gpt-4o-mini"
    "groq/llama-3.1-70b-versatile"
)

for llm in "${llms[@]}"
do
    echo "Running benchmark with LLM: $llm"
    python src/OntoRAG/benchmark/biomed.py \
        --method 'ontorag-simple' \
        --ontology_path data/ontologies/SNOMED \
        --llm "$llm" \
        --temperature 0.01 \
        --max_tokens 512
    echo "Finished running benchmark with LLM: $llm"
    echo "----------------------------------------"
done