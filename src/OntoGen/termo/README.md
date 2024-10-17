# TERMO: TERM ExtractiOn from Scientific Literature

TERMO is a tool for in-context extraction of _terms_, _acronyms_, _definitions_, and _relationships_ from scientific literature using Large Language Models (LLMs).

![Pipeline](docs/vocab.png)

# 1. Requirements

- __Ollama__ : TERMO is built using Ollama, a tool for running inference on LLMs. See [Ollama](https://ollama.com/) for instalation instructions.

# 2. Installation and Setup

```bash
pip install -r requirements.txt
```

Termo uses trasnformer model ```en_core_web_trf``` from Scapy which should be downloaded beforehand:

```bash
python -m spacy download en_core_web_trf
```

If instead of Ollama you want to use TERMO with [Anthropic API](https://docs.anthropic.com/en/api/getting-started), you need to setup first your Anthropic API key:

```bash
export ANTHROPIC_API_KEY=<YOUR_API_KEY>
```

# 3. Usage

## 3.1. Term Extraction

```python
my_text = "The Calvin cycle, light-independent reactions, bio synthetic phase..."

termo = Termo(my_text)
terms = termo.extract_terms(model="llama3.1:70b")
```

## 3.2. Acronym Extraction

```python
termo['terms'] = terms  # terms extracted in the first step
acronyms = termo.extract_acronyms(model="llama3.1:70b")
```

## 3.3. Definition Extraction

```python
termo['terms'] = terms  # terms extracted in the first step
definitions = termo.extract_definitions(model="llama3.1:70b")
```

## 3.4. Relationship Extraction

```python
termo['terms'] = terms  # terms extracted in the first step
relationships = termo.extract_relationships(model="llama3.1:70b")
```

