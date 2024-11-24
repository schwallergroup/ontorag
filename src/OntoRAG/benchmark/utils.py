"""Utils for dspy modules."""

import functools
from typing import Any, Callable, Optional

import dspy


def qa_decorator(reasoning: bool = False):
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                result = func(*args, **kwargs)
                return result

            except Exception as e:
                print(f"An error occurred: {str(e)}")
                out = dspy.Prediction(context=None, choice_answer=None)
                if reasoning:
                    out.reasoning = None
                return out

        return wrapper

    return decorator
