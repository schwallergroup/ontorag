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
from dspy.evaluate import Evaluate
from csvdatasets import CSVDataset


NUM_THREADS = 4
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
    # TODO Setup retrieval model
    rm = lambda x, k : [""] * k
    dspy.settings.configure(lm=llm, rm=rm)

def load_biomed_benchmarks():
    fpath = 'data/benchmarks/biomedical/'
    dfs = {}
    for fname in os.listdir(fpath):
        if fname.endswith('.csv') and not fname.startswith('.'):
            name = fname.strip('_qs.csv')
            dfs[name] = CSVDataset(fpath + fname, name, input_keys=['qprompt'])
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

def acc_metric_clean(gt, pred, trace=None):
    """Clean answer and compare with groundtruth."""
    return gt.answer == clean_model_answer(pred.choice_answer)

def run_benchmark(rag: dspy.Module, df, run):
    """Evaluate the method 'rag' on a dataset 'df'."""
    evaluate_program = Evaluate(
        devset=df.dev,
        metric=acc_metric_clean,
        num_threads=NUM_THREADS,
        display_progress=True,
        provide_traceback=True,
    )

    acc, results = evaluate_program(rag, return_outputs=True)
    ans = [{**s.toDict()} for r in results for s in r[:2]]

    table = wandb.Table(dataframe=pd.DataFrame(ans))
    run.log({f'accuracy_{df.name}': acc})
    run.log({f'results_{df.name}': table})

@contextmanager
def wandb_config(config: dict = None):
    try:
        with wandb.init(project='OntoRAG-biomed', config=config) as run:
            if config:
                wandb.config.update(config)
            yield run
    finally:
        pass

def run_one_method(method, ontology_path, llm, dfs, **kwargs):
    with wandb_config(config=dict(
        method=method,
        ontology_path=ontology_path,
        llm=llm,
        **kwargs
    )) as run:
        orag = METHODS[method](ontology_path=ontology_path, context='')
        for df in dfs.values():
            run_benchmark(orag, df, run)

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
    parser.add_argument('--method', type=str, default='ontorag-simple', choices=['all', *METHODS.keys()])
    parser.add_argument('--ontology_path', type=str, default='data/test/ontologies/SNOMED')
    parser.add_argument('--llm', type=str, default='gpt-4-turbo')
    parser.add_argument('--temperature', type=float, default=0.05)
    parser.add_argument('--max_tokens', type=int, default=512)
    args = parser.parse_args()
    main(**args.__dict__)