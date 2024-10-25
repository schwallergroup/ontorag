

# OntoGen

## Installation

## Example

```python
python extract_plain_text.py  --recompute  --no-skipping  --model='0.1.0-base'  --batchsize=10  docs/2304.05376.pdf
```

```python
python extract_sections.py --nougat  --abstract --introduction docs/2304.05376.processed.nougat.txt
python extract_sections.py --pymupdf  --abstract --introduction docs/2304.05376.processed.pymupdf.txt
```
```python
python run_termo.py llama3.1:70b docs/2304.05376.processed.nougat.introduction.txt --max_length_split_terms 2000 --max_length_split_definitions 10000 --max_length_split_relationships 10000 --num_ctx 9000 --temperature 0.9 
```
