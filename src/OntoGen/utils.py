import csv
from pathlib import Path
from unidecode import unidecode


def write_list(file, list_pdfs):
    with open(file, 'w') as f:
        for item in list_pdfs:
            f.write("%s\n" % item)


def read_list(file):
    with open(file, 'r') as f:
        list_pdfs = f.readlines()
        list_pdfs = [x.strip() for x in list_pdfs]
    return list_pdfs


def write_text(file, text):
    with open(file, 'w+') as f:
        f.write(text)


def read_text(file):
    try:
        with open(file, 'r') as f:
            text = f.read()
        return text
    except UnicodeDecodeError:
        with open(file, 'r', encoding='utf-8') as f:
            text = f.read()
        text = unidecode(text)
        return text


def write_tuples_list_to_csv(file_path, list_of_tuples):
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w+', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        for row in list_of_tuples:
            csvwriter.writerow(row)


def read_tuples_list_from_csv(file_path):
    file_path = Path(file_path)
    with open(file_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        list_of_tuples = [tuple(row) for row in csvreader]
    return list_of_tuples


def rename_file(file, new_name):
    file = Path(file)
    new_name = Path(new_name)
    file.rename(new_name)
    return new_name