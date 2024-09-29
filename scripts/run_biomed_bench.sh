#!/bin/bash

python src/OntoRAG/benchmark/biomed.py \
    --method all \
    --ontology_path data/test/ontologies/SNOMED \
    # --llm anthropic/claude-3-5-sonnet-20240620 \
    --llm mistral/mistral-medium-latest \
    --temperature 0.01 \
    --max_tokens 512
