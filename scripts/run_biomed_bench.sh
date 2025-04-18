#!/bin/bash

llms=(
    # "anthropic/claude-3-5-sonnet-20240620"
    # "anthropic/claude-3-sonnet-20240229"
    # "openai/gpt-4o"
    "openai/gpt-4o-mini"
    # "groq/llama-3.1-70b-versatile"
    # "mistral/mistral-medium-latest"
    # "openai/gpt-3.5-turbo"
    # "openai/gpt-4-turbo-preview"
)

run_benchmark() {
    local llm=$1
    echo "Running benchmarks for LLM: $llm"
    # Set rate limit depending on provider
    if [[ $llm == *"mistral"* ]]; then
        num_threads=4
    else
        num_threads=30
    fi
    # Meditron uses T=0.8 for all evals in all llms
    python src/OntoRAG/benchmark/biomed.py \
        --method all \
        --ontology_path data/ontologies/all/ \
        --llm "$llm" \
        --temperature 0.01 \
        --max_tokens 1024 \
        --num_threads "$num_threads"
    echo "Finished benchmarks for LLM: $llm"
    echo "----------------------------------------"
}

# Run benchmarks for each LLM in parallel
for llm in "${llms[@]}"
do
    run_benchmark "$llm" &
done

# Wait for all parallel processes to finish
wait

echo "All benchmarks completed."