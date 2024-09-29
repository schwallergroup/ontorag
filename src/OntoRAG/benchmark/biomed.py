"""Benchmarking OntoRAG on biomedical datasets."""

import os
import pandas as pd
from orag import *
from baselines import *
import dspy
from dotenv import load_dotenv
from contextlib import contextmanager
import argparse
import wandb


answer_col = {
    "mmlumed": lambda x: x['answer'],
    "medqa": lambda x: x['answer_idx'],
    "medmcqa": lambda x: x['answer'].replace({i+1:v for i,v in enumerate(['A', 'B', 'C', 'D'])}),
}

METHODS = {
    'ontorag-simple': SimpleORAG,
    'ontorag-hypo_ans': HyQORAG,
    'rag-zeroshot': QAZeroShot,
    'rag-context': QAContext,
    'rag-reason': QAReason,
    'rag-full': QAFull,
}


def init_dspy(llm: str = 'gpt-4-turbo', **kwargs):
    load_dotenv()
    llm = dspy.OpenAI(
        model=llm,
        temperature=kwargs.get('temperature', 0.5),
        max_tokens=kwargs.get('max_tokens', 512),
    )
    dspy.settings.configure(lm=llm)

def load_biomed_benchmarks():
    fpath = 'data/benchmarks/biomedical/'
    dfs = {}
    for fname in os.listdir(fpath):
        if fname.endswith('.csv') and not fname.startswith('.'):
            dfs[fname.strip('_qs.csv')] = pd.read_csv(fpath + fname)
    return dfs

def clean_model_answer(model_out):
    """Process the model output to get the answer."""
    inter = set(model_out).intersection(set('ABCDE'))
    if len(model_out) == 1:
        if model_out in list('ABCDE'):
            return model_out
    elif inter: 
        if len(inter) == 1:
            return inter.pop()
        elif ":" in model_out:
            if model_out.split(":")[0] in 'ABCDE':
                return model_out.split(":")[0]
    return None

def orag_wrap_series(orag):
    """Wrap calls to dspy modules into a pandas series."""
    def wrapper(x):
        results, context = orag(x)
        return pd.Series(dict(results=results, context=context))
    return wrapper

def run_benchmark(rag: dspy.Module, df, df_name):
    model_ans = df['qprompt'].apply(orag_wrap_series(rag))

    if 'reasoning' in model_ans['results'][0]:
        df['reasoning'] = model_ans['results'].apply(lambda x: x['reasoning'])
    else:
        df['reasoning'] = None
    df['raw_model_ans'] = model_ans['results'].apply(lambda x: x['choice_answer'])
    df['context_used'] = model_ans['context']
    df['model_answer'] = df['raw_model_ans'].apply(lambda x: clean_model_answer(x))

    acc = (df['model_answer'] == answer_col[df_name](df)).mean()
    return df, acc

@contextmanager
def wandb_config(config: dict = None):
    try:
        with wandb.init(project='OntoRAG-biomed', config=config) as run:
            if config:
                wandb.config.update(config)
            yield run
    finally:
        pass

def log_results(df, name, acc, run):
    df['gt_ans'] = answer_col[name](df)
    table = wandb.Table(dataframe=df[['qprompt', 'reasoning', 'gt_ans', 'model_answer', 'context_used']])
    run.log({f'accuracy_{name}': acc})
    run.log({f'results_{name}': table})

def run_one_method(method, ontology_path, llm, dfs, **kwargs):
    with wandb_config(config=dict(
        method='ontorag-simple',
        ontology_path=ontology_path,
        llm=llm,
        **kwargs
    )) as run:
        orag = METHODS[method](ontology_path=ontology_path, context='')
        for name, df in dfs.items():
            df = df.head(2) # For testing
            df, acc = run_benchmark(orag, df, name)
            log_results(df, name, acc, run)

def main(
        method: str = 'ontorag-simple',
        ontology_path: str = 'data/test/ontologies/SNOMED',
        llm: str = 'gpt-4-turbo',
        **kwargs
    ):
    init_dspy(llm = llm, **kwargs)
    dfs = load_biomed_benchmarks()
    if method == 'all':
        for method in METHODS.keys():
            run_one_method(method, ontology_path, llm, dfs, **kwargs)
    elif method in METHODS:
        run_one_method(method, ontology_path, llm, dfs, **kwargs)
    else:
        raise ValueError(f"Method {method} not found.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='all', choices=['all', 'ontorag-simple', 'ontorag-hypo_ans'])
    parser.add_argument('--ontology_path', type=str, default='data/test/ontologies/SNOMED')
    parser.add_argument('--llm', type=str, default='gpt-4-turbo')
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--max_tokens', type=int, default=512)
    args = parser.parse_args()
    main(**args.__dict__)