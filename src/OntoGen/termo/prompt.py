prompt_abstract = '''
Given this scientific text:

===
{text}
===

Extract every term or noun phrase from the scientific text, including both single-word terms, compound terms, section titles and terms involving multiple words. Ensure to capture all specific technical terms, acronyms, and units. Do not include non-scientific terms such as author names, journal names, institutions, editorial information or any publishing related data (e.g. 'published date'). Output a list that starts with '-'.
'''


#Given the following context and the following vocabulary, identify all the acronyms and symbols in the vocabulary and their corresponding terms, according to the context. 
prompt_acronym = '''
Given the following context and the following vocabulary, identify all the acronyms and symbols in the context and their corresponding terms in the vocabulary, according to the context. 
Do not print anything else other than the acronyms and their corresponding terms.
Provide the results in the format:

<acronym1>: <term1>
<acronym2>: <term2>

Context:

===
{CONTEXT}
===

Vocabulary:

===
{VOCABULARY}
===
'''


prompt_verbs = '''
Given this scientific text:

===
{CONTEXT}
===

Extract every verb from the scientific text, including both single-word verbs and composed verbs. Ensure to capture all specific technical verbs. Do not include non-scientific verbs such as verbs related to authors, journals, institutions, editorial information or any publishing related data (e.g. 'published'). Output a list that starts with '-'.
'''


prompt_triplets = '''
Given this scientific text:
===
{CONTEXT}
===
This vocabulary:
===
{VOCABULARY}
===
Extract every triplet (term > relationship > term) from the scientific text that involves the terms presented in the vocabulary. Extract only triplets that explicitely appear in the text. Use only the terms presented in the vocabulary. Focus only in short relationships, with around five or less words. Do not ouput long relationships. Output a list that starts with '-'.
'''

prompt_definitions = '''
Given this scientific text:
===
{CONTEXT}
===
This vocabulary:
===
{VOCABULARY}
===
Extract the definition of each term in the vocabulary from the scientific text. For the definitions, use only the context provided, this is, do not include any informaiton in the definitions that is not present in the provided text. Focus only in short, brief and very concise definitions. Do not ouput long definitions. If no explicit definition is provided, provide the information available in the context to undestand the meaning of the term. Output a list that starts with '-'.
'''