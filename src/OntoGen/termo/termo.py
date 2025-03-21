import os
import re
from functools import cmp_to_key

import anthropic
import ollama
import spacy
from prompt import (
    prompt_abstract,
    prompt_acronym,
    prompt_definitions,
    prompt_triplets,
)
from thinc.api import require_gpu, set_gpu_allocator


class Termo(dict):
    def __init__(
        self,
        text,
        remove_duplicates=True,
        remove_substrings=True,
        backend="ollama",
    ):
        """
        text: str
            The text from which to extract terms, acronyms, relationships and definitions
        remove_duplicates: bool
            Whether to remove duplicated terms from the list of identified terms
        remove_substrings: bool
            Whether to remove terms that are substrings of other terms
        backend: str
            The backend to use for the queries. Available backends are 'ollama' and 'anthropic'
        """
        self.text = text
        self.remove_duplicates = remove_duplicates
        self.remove_substrings = remove_substrings
        self["terms"] = []
        self["acronyms"] = {}
        if backend == "ollama":
            self.query_fn = self.query_ollama
        elif backend == "anthropic":
            self.query_fn = self.query_anthropic
        else:
            raise ValueError(
                f"Unknown backend: '{backend}'. Available backends are 'ollama' and 'anthropic'"
            )

    def extract_terms(
        self, model, space_separator=True, max_length_split=2000, **kwargs
    ):
        """
        Extract terms from the text using the specified model
        model: str
            The model to use for the queries
        space_separator: bool
            If True, only terms separated by spaces are considered. If False, any match of the given term in the text is considered,
            even if its not surrounded by spaces.
        max_length_split: int
            The maximum length of the text to send in each query in characters
        """
        self["terms"] += self.get_filtered_list_from_llm(
            model, self.text, space_separator, max_length_split, **kwargs
        )
        self["terms"] = self.postprocess_terms(
            self["terms"], self.remove_duplicates, self.remove_substrings
        )
        return self["terms"]

    def extract_acronyms(self, model, max_length_split=2000, **kwargs):
        terms = [term[0] for term in self["terms"]]
        self["acronyms"] = self.get_acronyms_from_llm(
            model, self.text, terms, max_length_split, **kwargs
        )
        self["acronyms"] = self.postprocess_acronyms(
            self["acronyms"], self.text, terms
        )
        return self["acronyms"]

    def extract_relationships(self, model, max_length_split=2000, **kwargs):
        terms = [term[0] for term in self["terms"]]
        acronyms = [ac[0] for ac in self["acronyms"].items()]
        # here, exclusively, we consider acronyms as terms
        terms += acronyms
        self["relationships"] = self.get_relationships_from_llm(
            model, self.text, terms, max_length_split, **kwargs
        )
        self["relationships"] = self.postprocess_relationships(
            self["relationships"], self.text, terms
        )
        return self["relationships"]

    def extract_definitions(self, model, max_length_split=2000, **kwargs):
        terms = [term[0] for term in self["terms"]]
        self["definitions"] = self.get_definitions_from_llm(
            model, self.text, terms, max_length_split, **kwargs
        )
        self["definitions"] = self.postprocess_definitions(
            self["definitions"], self.text, terms
        )
        return self["definitions"]

    def query_anthropic(self, model, prompt, **kwargs):
        client = anthropic.Anthropic(
            api_key=os.environ["ANTHROPIC_API_KEY"],
        )
        message = client.messages.create(
            model=model,
            max_tokens=8000,
            temperature=0.3,
            messages=[
                {"role": "user", "content": [{"type": "text", "text": prompt}]}
            ],
        )
        return message.content[0].text

    def query_ollama(self, model, prompt, **kwargs):
        response = ollama.generate(model=model, prompt=prompt, **kwargs)
        return response["response"]

    def postprocess_terms(
        self, all_terms, remove_duplicates=True, remove_substrings=True
    ):
        if remove_duplicates:
            all_terms = self.remove_duplicated_terms(all_terms)
        if remove_substrings:
            all_terms = self.remove_substrings_from_list(all_terms)
        return all_terms

    def postprocess_relationships(self, relationships, text, terms):
        # remove those relationships with terms not in terms list
        result = []
        lower_terms = [term.lower() for term in terms]
        for t1, rel, t2 in relationships:
            if t1.lower() in lower_terms and t2.lower() in lower_terms:
                result.append((t1, rel, t2))
            else:
                print(
                    f"Removing relationship '{t1} > {rel} > {t2}' because it contains terms not in the text"
                )
        return result

    def postprocess_definitions(self, definitions, text, terms):
        # remove those relationships with terms not in terms list
        result = {}
        lower_terms = [term.lower() for term in terms]
        for term, defi in definitions.items():
            if term.lower() in lower_terms:
                result[term] = defi
            else:
                print(
                    f"Removing definitions for '{term}' because unknown term"
                )
        return result

    def remove_duplicated_terms(self, all_terms):
        return list(set(all_terms))

    def remove_substrings_from_list(self, all_terms):
        # sort by increasing end position. If two terms have the same end position, sort by increasing start position
        def comp(x, y):
            if x[1] != y[1]:  # compare start position
                return x[1] - y[1]
            return y[2] - x[2]  # compare end position

        all_terms = sorted(all_terms, key=cmp_to_key(comp))

        # remove all terms that are substrings of other terms
        if len(all_terms) == 0:
            return []
        start, end = all_terms[0][1], all_terms[0][2]
        result = [all_terms[0]]
        for i in range(1, len(all_terms)):
            if all_terms[i][1] >= start and all_terms[i][2] <= end:
                print(
                    f"Removing term {all_terms[i][0]} because it is a substring of {all_terms[i-1][0]}"
                )
                continue
            start, end = all_terms[i][1], all_terms[i][2]
            result.append(all_terms[i])

        return result

    def split_text_into_lines(self, text):
        set_gpu_allocator("pytorch")
        require_gpu(0)

        nlp = spacy.load("en_core_web_trf")
        doc = nlp(text)

        sentences = [sent.text for sent in doc.sents]
        return sentences

    def get_list_from_llm(self, model, text, max_length_split=2000, **kwargs):

        lines = self.split_text_into_lines(text)

        list_terms = []
        # build chunks of text with a maximum length of max_length_split
        chunks = []
        current_chunk = ""
        for line in lines:
            line = line.strip()
            if len(current_chunk) + len(line) > max_length_split:
                chunks.append(current_chunk)
                current_chunk = ""
            # sometimes a multiple word term is split into multiple lines so a space is added (instead of a newline)
            current_chunk = current_chunk + " " + line
        if len(current_chunk) > 0:
            chunks.append(current_chunk)

        for chunk in chunks:
            if len(chunk.replace("\n", "").strip()) == 0:
                continue
            prompt = prompt_abstract.format(text=chunk)
            response = self.query_ollama(model, prompt, **kwargs)
            for l in response.split("\n"):
                if len(l) > 0 and l[0] == "-" and len(l[1:].strip()) > 0:
                    list_terms.append(l[1:].strip())

        return list_terms

    def _build_chunks(self, text, max_length_split):
        lines = self.split_text_into_lines(text)
        chunks = []
        current_chunk = ""
        for line in lines:
            line = line.strip()
            if len(current_chunk) + len(line) > max_length_split:
                chunks.append(current_chunk)
                current_chunk = ""
            current_chunk = current_chunk + " " + line
        if len(current_chunk) > 0:
            chunks.append(current_chunk)
        return chunks

    def get_relationships_from_llm(
        self, model, text, terms, max_length_split=2000, **kwargs
    ):

        relationships = []
        set_terms = set(terms)

        chunks = self._build_chunks(text, max_length_split)

        for chunk in chunks:
            if len(chunk.strip()) == 0:
                continue
            prompt = prompt_triplets.format(
                CONTEXT=chunk, VOCABULARY="\n".join(set_terms)
            )
            response = self.query_ollama(model, prompt, **kwargs)

            for l in response.split("\n"):
                if len(l) > 0 and l[0] == "-" and len(l[1:].strip()) > 0:
                    l = l[1:].strip()
                    split = l.split(">")
                    if len(split) != 3:
                        continue
                    t1, rel, t2 = (
                        split[0].strip(),
                        split[1].strip(),
                        split[2].strip(),
                    )
                    t1, rel, t2 = (
                        t1.replace("*", "").strip(),
                        rel.replace("*", "").strip(),
                        t2.replace("*", "").strip(),
                    )

                    if len(t1) == 0 or len(rel) == 0 or len(t2) == 0:
                        continue
                    relationships.append((t1, rel, t2))
        return relationships

    def get_definitions_from_llm(
        self, model, text, terms, max_length_split=2000, **kwargs
    ):

        definitions = {}
        set_terms = set(terms)

        chunks = self._build_chunks(text, max_length_split)
        for chunk in chunks:
            if len(chunk.strip()) == 0:
                continue
            prompt = prompt_definitions.format(
                CONTEXT=chunk, VOCABULARY="\n".join(set_terms)
            )
            response = self.query_ollama(model, prompt, **kwargs)

            for l in response.split("\n"):
                if (
                    len(l) > 0
                    and l[0] == "-"
                    and len(l[1:].strip()) > 0
                    and "-" in l
                ):
                    l = l[1:].strip()  # remove initial '-'
                    split = l.split(":")
                    if len(split) != 2:
                        continue
                    term, defi = split[0].strip(), split[1].strip()
                    term, defi = (
                        term.replace("*", "").strip(),
                        defi.replace("*", "").strip(),
                    )

                    if len(term) == 0 or len(defi) == 0:
                        continue
                    definitions[term] = defi

        return definitions

    def get_acronyms_from_llm(
        self, model, text, terms, max_length_split=2000, **kwargs
    ):

        acronyms = {}
        set_terms = set(terms)

        lines = self.split_text_into_lines(text)

        # build chunks of text with a maximum length of max_length_split
        chunks = []
        current_chunk = ""
        for line in lines:
            line = line.strip()
            if len(current_chunk) + len(line) > max_length_split:
                chunks.append(current_chunk)
                current_chunk = ""
            current_chunk += line
        if len(current_chunk) > 0:
            chunks.append(current_chunk)

        for chunk in chunks:
            if len(chunk.strip()) == 0:
                continue
            prompt = prompt_acronym.format(
                CONTEXT=chunk, VOCABULARY="\n".join(set_terms)
            )
            response = self.query_ollama(model, prompt, **kwargs)
            acronyms = {}
            for l in response.split("\n"):
                if len(l) > 0:
                    split = l.split(":")
                    if len(split) != 2:
                        continue
                    acronym, term = split[0].strip(), split[1].strip()
                    acronym, term = (
                        acronym.replace("*", "").strip(),
                        term.replace("*", "").strip(),
                    )
                    if len(acronym) == 0 or len(term) == 0:
                        continue
                    acronyms[acronym] = term
        return acronyms

    def get_acronyms_from_llm_full_text(self, model, text, terms, **params):
        set_terms = set(terms)
        prompt = prompt_acronym.format(
            CONTEXT=text, VOCABULARY="\n".join(set_terms)
        )
        response = self.query_ollama(model, prompt, **params)
        acronyms = {}

        for l in response["response"].split("\n"):
            if len(l) > 0:
                split = l.split(":")
                if len(split) != 2:
                    continue
                acronym, term = split[0].strip(), split[1].strip()
                if len(acronym.strip()) == 0 and len(term.strip()) == 0:
                    continue
                acronym, term = (
                    acronym.replace("*", "").strip(),
                    term.replace("*", "").strip(),
                )
                acronyms[acronym] = term
        return acronyms

    def postprocess_acronyms(self, acronyms, text, terms):
        # remove acronyms that are not in the text
        result = {}
        for acronym, term in acronyms.items():
            if (
                term.lower() in text.lower()
                and acronym.lower() in text.lower()
            ):
                result[acronym] = term
            else:
                print(
                    f"Removing acronym '{acronym}':'{term}' because it is not in the text"
                )
        return result

    def compute_matches_without_spaces(self, term, text):
        """
        Removes spaces from both the input term and input text and tries to find matches.
        The positions returned are with respect to the original text.
        All the text is converted to lower case.
        It is assumed that no match can happen in a substring of a term.
        """
        lower_text_no_spaces = text.lower().replace(" ", "")
        lower_term_no_spaces = term.lower().replace(" ", "")
        matches = [
            (m.start(), m.end(), term)
            for m in re.finditer(
                re.escape(lower_term_no_spaces), lower_text_no_spaces
            )
        ]
        if len(matches) == 0:
            return []
        matches_with_spaces = []
        matches.sort(key=lambda x: x[0])
        index_matches = 0
        index_no_space = 0
        for index_spaces in range(len(text)):
            if index_no_space == matches[index_matches][0]:
                matches_with_spaces.append(
                    (index_spaces, index_spaces + len(term), term)
                )
                index_matches += 1
                if index_matches == len(matches):
                    break
            if text[index_spaces] != " ":
                index_no_space += 1
        return matches_with_spaces

    def get_filtered_list_from_llm(
        self,
        model,
        text,
        space_separator=True,
        max_length_split=2000,
        **kwargs,
    ):
        terms = self.get_list_from_llm(model, text, max_length_split, **kwargs)

        sentences = self.split_text_into_lines(text)

        # remove hallucinated terms
        all_terms = []
        for term in terms:
            term_matches = []
            c = 0  # sentence length accumulator
            for i, sentence in enumerate(sentences):
                sentence = sentence.lower()
                matches_start_space = [
                    (term, c + m.start() + 1, c + m.end(), i)
                    for m in re.finditer(
                        re.escape(" " + term.lower()), sentence
                    )
                ]
                matches_end_space = [
                    (term, c + m.start(), c + m.end() - 1, i)
                    for m in re.finditer(
                        re.escape(term.lower() + " "), sentence
                    )
                ]
                matches_start_dot = [
                    (term, c + m.start() + 1, c + m.end(), i)
                    for m in re.finditer(
                        re.escape("." + term.lower()), sentence
                    )
                ]
                matches_end_dot = [
                    (term, c + m.start(), c + m.end() - 1, i)
                    for m in re.finditer(
                        re.escape(term.lower() + "."), sentence
                    )
                ]
                matches_start_comma = [
                    (term, c + m.start() + 1, c + m.end(), i)
                    for m in re.finditer(
                        re.escape("," + term.lower()), sentence
                    )
                ]
                matches_end_comma = [
                    (term, c + m.start(), c + m.end() - 1, i)
                    for m in re.finditer(
                        re.escape(term.lower() + ","), sentence
                    )
                ]
                matches_start_paren = [
                    (term, c + m.start() + 1, c + m.end(), i)
                    for m in re.finditer(
                        re.escape("(" + term.lower()), sentence
                    )
                ]
                matches_end_paren = [
                    (term, c + m.start(), c + m.end() - 1, i)
                    for m in re.finditer(
                        re.escape(term.lower() + ")"), sentence
                    )
                ]
                matches_no_space = [
                    (term, c + m.start(), c + m.end(), i)
                    for m in re.finditer(re.escape(term.lower()), sentence)
                ]
                # NOTE: we skip the no-space matching as it potentially creates too many false positives
                # matches_no_space_text = compute_matches_without_spaces(term, sentence)
                # matches = matches_start_space + matches_end_space + matches_no_space_text
                matches = (
                    matches_start_space
                    + matches_end_space
                    + matches_start_dot
                    + matches_end_dot
                    + matches_start_comma
                    + matches_end_comma
                    + matches_start_paren
                    + matches_end_paren
                )
                if not space_separator:
                    matches += matches_no_space
                # remove duplicates from matches
                matches = self.remove_duplicated_terms(matches)
                term_matches += matches
                c += len(sentence) + 1  # +1 for the periods
            if len(term_matches) == 0:
                print(f"Term '{term}' not found in text")
            all_terms += term_matches

        return all_terms
