import subprocess
import pymupdf
import argparse
import re
from pathlib import Path
from nougat.postprocessing import postprocess_single
from utils import write_text, rename_file
from unidecode import unidecode


def get_pymupdf_text(pdf_path):
    '''
    Extract text from a pdf file using pymupdf. The method also removes unicode characters.

    Parameters:
    pdf_path (str): Path to the pdf file.
    '''
    doc = pymupdf.open(pdf_path)

    text = ''
    for page in doc:
        text += page.get_text()
    text = unidecode(text)
    return text


def postprocess_pymupdf(text):
    '''
    Postprocess the text extracted from a pdf file using pymupdf.
    This includes removing empty lines, removing multiple \n, and removing lines with only one character.

    Parameters:
    text (str): Text extracted from a pdf file using pymupdf.
    '''
    # Remove lines with only one character, which are common in 
    # elsevier papers, delimiting each page with the name of the journal
    lines = text.split('\n')
    lines = [l for l in lines if len(l) > 1]
    text = '\n'.join(lines)

    # remove multiple \n
    while '\n\n' in text:
        text = text.replace('\n\n', '\n')
    
    return text


def extract_doi(txt_file):
    '''
    Extract the DOI from a text file using a regular expression.

    Parameters:
    txt_file (str): Path to the text file.
    '''
    path = Path(txt_file)
    text = path.read_text()
    # return the first match only
    match = re.search(r'\b(10[.][0-9]{4,}(?:[.][0-9]+)*/(?:(?!["&\'<>])\S)+)\b', text)
    if match:
        return match.group(0)
    return None


def nougat_process_pdf(pdf_path, batchsize=None, model=None, out=None, recompute=False, full_precision=False, markdown=True, no_skipping=None, pages=None):
    '''
    --batchsize BATCHSIZE, -b BATCHSIZE
                            Batch size to use.
    --checkpoint CHECKPOINT, -c CHECKPOINT
                            Path to checkpoint directory.
    --model MODEL_TAG, -m MODEL_TAG
                            Model tag to use.
    --out OUT, -o OUT     Output directory.
    --recompute           Recompute already computed PDF, discarding previous predictions.
    --full-precision      Use float32 instead of bfloat16. Can speed up CPU conversion for some setups.
    --no-markdown         Do not add postprocessing step for markdown compatibility.
    --markdown            Add postprocessing step for markdown compatibility (default).
    --no-skipping         Don't apply failure detection heuristic.
    --pages PAGES, -p PAGES
                            Provide page numbers like '1-4,7' for pages 1 through 4 and page 7. Only works for single PDFs.
    '''
    args = ["nougat"]
    if batchsize:
        args.extend(["--batchsize", str(batchsize)])
    if model:
        args.extend(["--model", model])
    if out:
        args.extend(["--out", out])
    if recompute:
        args.append("--recompute")
    if full_precision:
        args.append("--full-precision")
    if not markdown:
        args.append("--no-markdown")
    if no_skipping:
        args.append("--no-skipping")
    if pages:
        args.extend(["--pages", pages])
    args.append(pdf_path)
    output = subprocess.run(args, capture_output=True)
    return output.stderr.decode("utf-8")


def process_pdf(path_pdf, threshold_num_repetitions=5, **params):
    '''
    Extract plain text and postprocess it from a pdf file using both Nougat and PyMuPDF.
    This includes removing repeated lines, footnotes, and multiple \n.
    The method also extracts the DOI from the text extracted using PyMuPDF.

    Parameters:
    path_pdf (str): Path to the pdf file.
    threshold_num_repetitions (int): Threshold number of repetitions to remove repeated lines.
    **params: Additional parameters to pass to the nougat extraction method.
    '''
    p = Path(path_pdf)
    if p.suffix != '.pdf':
        raise ValueError(f"File {p} is not a pdf file.")
    
    raw_output_nougat_file = p.parent / (p.stem + '.raw.nougat.txt')
    raw_pymupdf_file = p.parent / (p.stem + '.raw.pymupdf.txt')
    processed_output_nougat_file = p.parent / (p.stem + '.processed.nougat.txt')
    processed_output_pymupdf_file = p.parent / (p.stem + '.processed.pymupdf.txt')
    doi_file = p.parent / (p.stem + '.doi.txt')

    ## Nougat
    err = nougat_process_pdf(p, out=p.parent, **params)
    print(err)
    rename_file(p.parent / (p.stem + '.mmd'), raw_output_nougat_file)
    if not raw_output_nougat_file.exists():
        raise ValueError(f"File {raw_output_nougat_file} not found. Probably the nougat processing failed.")
    
    raw_text = raw_output_nougat_file.read_text()
    raw_text = remove_repeated_lines(raw_text, threshold_num_repetitions=threshold_num_repetitions)
    raw_text = remove_footnotes(raw_text)
    raw_text = remove_multiple_jumplines(raw_text)
    write_text(processed_output_nougat_file, raw_text)

    ## PyMuPDF
    raw_pymupdf_text = get_pymupdf_text(p)
    write_text(raw_pymupdf_file, raw_pymupdf_text)
    processed_pymupdf_text = postprocess_pymupdf(raw_pymupdf_text)
    write_text(processed_output_pymupdf_file, processed_pymupdf_text)

    # Doi
    doi = extract_doi(raw_pymupdf_file)
    if doi:
        write_text(doi_file, doi)


def remove_repeated_lines(text, threshold_num_repetitions=10):
    '''
    Remove repeated lines from a text.

    Parameters:
    text (str): Text to process.
    threshold_num_repetitions (int): Threshold number of repetitions to remove repeated lines.
    '''
    lines = text.split('\n')
    for line in lines:
        if len(line.strip()) == 0:
            continue
        count = 0
        for l in lines:
            if l == line:
                count += 1
        if count > threshold_num_repetitions:
            text = text.replace(line, '')
    return text

     
def remove_multiple_jumplines(text):
    '''
    Remove \n\n\n from a text and replace it with \n\n.

    Parameters:
    text (str): Text to process.
    '''
    while '\n\n\n' in text:
        text = text.replace('\n\n\n', '\n\n')
    return text


def remove_footnotes(text):
    '''
    Remove lines starting with 'footnote' from a text.
    This is common in texts extracted with nougat.

    Parameters:
    text (str): Text to process.
    '''
    lines = text.split('\n')
    for line in lines:
        if line.lower().startswith('footnote'):
            text = text.replace(line, '')
    return text


def remove_long_lines(text, threshold=400):
    '''
    Remove lines longer than a threshold from a text.

    Parameters:
    text (str): Text to process.
    threshold (int): Threshold length to remove lines.
    '''
    lines = text.split('\n')
    filtered_lines = []
    for line in lines:
        if len(line) < threshold:
            filtered_lines.append(line)
    return '\n'.join(filtered_lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", help="Path to the pdf file to process")
    parser.add_argument("--batchsize", "-b", help="(Nougat) Batch size to use.", type=int)
    parser.add_argument("--model", "-m", help="(Nougat) model tag to use.")
    parser.add_argument("--recompute", help="(Nougat) Recompute already computed PDF, discarding previous predictions.", action="store_true")
    parser.add_argument("--full-precision", help="(Nougat) Use float32 instead of bfloat16. Can speed up CPU conversion for some setups.", action="store_true")
    parser.add_argument("--markdown", help="(Nougat) Add postprocessing step for markdown compatibility (default).", action="store_true")
    parser.add_argument("--no-skipping", help="(Nougat) Don't apply failure detection heuristic.", action="store_true")
    parser.add_argument("--pages", "-p", help="(Nougat) Provide page numbers like '1-4,7' for pages 1 through 4 and page 7. Only works for single PDFs.")
    parser.add_argument("--threshold_num_repetitions", help="(Nougat) Threshold number of repetitions to remove repeated lines.", type=int, default=5)
    args = parser.parse_args()
    args_pdf = args.pdf
    # remove args.pdf from args
    delattr(args, 'pdf')
    process_pdf(args_pdf, **vars(args))