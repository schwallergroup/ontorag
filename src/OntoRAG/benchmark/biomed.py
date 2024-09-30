"""Benchmarking OntoRAG on biomedical datasets."""

import argparse
import os

import dspy
import pandas as pd
from baselines import QAContext, QAFull, QAReason, QAZeroShot
from csvdatasets import CSVDataset
from dotenv import load_dotenv
from dspy.evaluate import Evaluate
from orag import HyQORAG, SimpleORAG, OntoRAGTM, HyQOntoRAGTM

import wandb
from OntoRAG.utils import OntoRetriever

METHODS = {
    # "ontorag-simple": SimpleORAG,
    # "ontorag-hypo_ans": HyQORAG,
    "ontorag-tm": OntoRAGTM,
    "ontorag-hypo_ans-tm": HyQOntoRAGTM,
    # "rag-zeroshot": QAZeroShot,
    # "rag-reason": QAReason,
    # 'rag-context': QAContext,
    # 'rag-full': QAFull,
}


def _init_dspy(llm: str = "gpt-4-turbo", **kwargs):
    load_dotenv()
    llm = dspy.LM(
        model=llm,
        temperature=kwargs.get("temperature", 0.5),
        max_tokens=kwargs.get("max_tokens", 512),
        max_retries=kwargs.get("max_retries", 3),
    )
    # TODO Setup retrieval model
    rm = lambda x, k: [""] * k
    dspy.settings.configure(lm=llm, rm=rm)


def _load_biomed_benchmarks():
    fpath = "data/benchmarks/biomedical/"
    dfs = {}
    for fname in os.listdir(fpath):
        if fname.endswith(".csv") and not fname.startswith("."):
            name = fname.strip("_qs.csv")
            dfs[name] = CSVDataset(fpath + fname, name, input_keys=["qprompt"])
    return dfs


def clean_model_answer(model_out):
    """Process the model output to get the answer."""
    inter = set(model_out).intersection(set("ABCDE"))
    if len(model_out) == 1:
        if model_out in list("ABCDE"):
            return model_out
    elif inter:
        if len(inter) == 1:
            return inter.pop()
        elif ":" in model_out:
            if model_out.split(":")[0] in "ABCDE":
                return model_out.split(":")[0]
    return None


def compile_results(results):
    """If model failed to answer, fill with empty prediction."""
    empty = dspy.Prediction(reasoning="", choice_answer="")
    cresults = [
        r if not isinstance(r[1], dict) else (r[0], empty) for r in results
    ]
    table = [{**r[0].toDict(), **r[1].toDict()} for r in cresults]
    return table


def acc_metric_clean(gt, pred, trace=None):
    """Clean answer and compare with groundtruth."""
    return gt.answer == clean_model_answer(pred.choice_answer)


def run_benchmark(rag: dspy.Module, df, run, num_threads=4):
    """Evaluate the method 'rag' on a dataset 'df'."""
    evaluate_program = Evaluate(
        devset=df.dev,
        metric=acc_metric_clean,
        num_threads=num_threads,
        display_progress=True,
    )

    acc, results = evaluate_program(rag, return_outputs=True)
    table = compile_results(results)
    table = wandb.Table(dataframe=pd.DataFrame(table))  # type: ignore
    run.log({f"accuracy_{df.name}": acc})
    run.log({f"results_{df.name}": table})


def run_one_method(method, ontology, llm, dfs, **kwargs):
    """Run one method on all datasets."""
    with wandb.init(
        project="OntoRAG-biomed",
        config=dict(
            method=method,
            ontology_path=(
                ontology.fpath
                if isinstance(ontology, OntoRetriever)
                else ontology
            ),
            llm=llm,
            **kwargs,
        ),
    ) as run:
        orag = METHODS[method](ontology=ontology, context="")
        for df in dfs.values():
            run_benchmark(
                orag, df, run, num_threads=kwargs.get("num_threads", 4)
            )


def main(
    method: str = "ontorag-simple",
    ontology_path: str = "data/test/ontologies/SNOMED",
    llm: str = "gpt-4-turbo",
    **kwargs,
):
    """Run evaluation on all benchmarks."""
    _init_dspy(llm=llm, **kwargs)
    dfs = _load_biomed_benchmarks()
    ontology = OntoRetriever(ontology_path=ontology_path)

    if method == "all":
        for method in METHODS.keys():
            run_one_method(method, ontology, llm, dfs, **kwargs)
    elif method in METHODS:
        run_one_method(method, ontology, llm, dfs, **kwargs)
    else:
        raise ValueError(f"Method {method} not found.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        type=str,
        default="ontorag-simple",
        choices=["all", *METHODS.keys()],
    )
    parser.add_argument(
        "--ontology_path", type=str, default="data/test/ontologies/SNOMED"
    )
    parser.add_argument(
        "--llm", type=str, default="anthropic/claude-3-5-sonnet-20240620"
    )
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--num_threads", type=int, default=4)
    args = parser.parse_args()
    main(**args.__dict__)
