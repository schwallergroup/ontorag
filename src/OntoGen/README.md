

# OntoGen

## Installation

## Example: from paper to taxonomy

```bash
python extract_plain_text.py  --recompute  --no-skipping  --model='0.1.0-base'  --batchsize=10  docs/2304.05376.pdf
```

```bash
python extract_sections.py --nougat  --abstract --introduction docs/2304.05376.processed.nougat.txt
python extract_sections.py --pymupdf  --abstract --introduction docs/2304.05376.processed.pymupdf.txt
```

```bash
python run_termo.py \
llama3.1:70b \
docs/2304.05376.processed.nougat.abstract.txt \
--temperature 0.9 \
--max_length_split_terms 2000 \
--max_length_split_definitions 10000 \
--max_length_split_relationships 10000 \
--num_ctx 9000
```

```bash
python run_termo.py \
llama3.1:70b \
docs/2304.05376.processed.nougat.introduction.txt \
--max_length_split_terms 2000 \
--max_length_split_definitions 10000 \
--max_length_split_relationships 10000 \
--num_ctx 9000 \
--temperature 0.9 
```

```bash
python generate_categories.py \
'Chemistry Augmented Language Models' \
docs/2304.05376.processed.nougat.abstract.txt docs/2304.05376.processed.nougat.introduction.txt \
--generation-model 'llama3.1:8b-instruct-fp16' \
--generation-temperature 0.5 \
--generation-num-ctx 16000 \
--format-model 'llama3.1:70b' \
--format-num-ctx 16000 \
--synthesis-model 'llama3.1:70b' \
--synthesis-num-ctx 16000 \
--num-retries 5 --num-generated-seed 5
```

```bash
python generate_taxonomy.py \
categories/Chemistry\ Augmented\ Language\ Models_categories_seed.0.txt \
docs/2304.05376.processed.nougat.abstract.txt \
docs/2304.05376.processed.nougat.introduction.txt \
--num-ctx 32000 \
--temperature 0.1 \ 
--model 'llama3.1:70b' \
```

![Output taxonomy](docs/output.png "Generated taxonomy")