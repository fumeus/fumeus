import csv, json
from fum_utils import *


def csv_col_to_list(dataset_fname, x_column, header = False):
    """
    Reads a csv file and returns the contents of that file as a list.

    dataset_fname - the file name for the dataset to read.
    header - whether the dataset contains a header row (Boolean).
     >> Default: False.

    Returns x (2-D list).
    """

    # Initialize data containers and check extension of provided file name
    x = []
    dataset_fname = check_extension(dataset_fname)

    # Extract column positioned at x_column
    encoding = guess_encoding(dataset_fname)
    with open(dataset_fname, "r", encoding = encoding) as file:
        reader = csv.reader(file, delimiter = ",")
        for i, line in enumerate(reader):
            if i > 0 or not header:
                x.append(line[x_column])

    return x


def main(dataset_fname, terms_fname, output_fname, header = False, header2 = False, x_column = 0, format = "csv"):
    """
    Main driver code to perform text analysis.
    Loads dataset from a CSV file.
    Performs text cleaning on loaded text data.
    Scores text based on loaded terms list.
    Outputs scored terms to CSV or JSON file.
    
    dataset_fname - the file name for the dataset to read.
    terms_fname - the file name for the terms/dictionary to read.
    output_fname - the file name for the output file.
    header - whether the dataset file contains a header row (Boolean).
     >> Default: False.
    header2 - whether the terms/dictionary file contains a header row (Boolean).
     >> Default: False.
    x_column - the index of the column to treat as x (textual) data in dataset_fname.
     >> Default: 0.
    top_n - length of each n-gram list to output to CSV files.
     >> Default: 200.
    format - the output format to utilize (either CSV or JSON).
     >> Default: "csv".
    """

    # Load data from a CSV file
    # x is the textual data
    x = csv_col_to_list(dataset_fname, x_column, header)

    # Clean text data
    x_clean = clean_documents(x)
    
    # Load terms list
    terms_matrix = read_terms_csv(terms_fname, header2)
    
    # Generate scores
    x_scored = [weighted_score_calculation(" ".join(record), terms_matrix) for record in x_clean]
    
    # Output scores to CSV or JSON files
    if format == "csv":
        scores_to_csv(x, x_scored, output_fname)
    elif format == "json":
        scores_to_json(x, x_scored, output_fname)


def read_terms_csv(fname, header = False):
    """
    Reads a list of terms from a CSV file.
    
    fname - the file name for the terms to read.
    header - whether the terms/dictionary file contains a header row (Boolean).
     >> Default: False.

    Returns a 2D list comprised of (term, weight).
    """
    
    table = []
    encoding = guess_encoding(fname)
    with open(fname, "r", encoding = encoding) as file:
        reader = csv.reader(file)
        for i, line in enumerate(reader):
            if i > 0 or not header:
                term = line[0]
                weight = float(line[1])
                table.append((term, weight))
                
    return table


def scores_to_csv(x, x_scored, fname):
    """
    Writes a the scored records to an output CSV file.
    
    x - all non-y columns.
    x_scored - score computations.
    fname - the file name for the output file.
    """
    
    fname = check_extension(fname)
    
    table = []
    for i, text in enumerate(x):
        table.append([x[i]] + x_scored[i])

    table = sorted(table, key = lambda l:l[-2], reverse = True)

    with open(fname, "w") as file:
        writer = csv.writer(file, lineterminator = "\n")
        for row in table:
            writer.writerow(row)


def scores_to_json(x, x_scored, fname):
    """
    Writes a the scored records to an output JSON file.
    
    x - all non-y columns.
    x_scored - score computations.
    fname - the file name for the output file.
    """
    
    fname = check_extension(fname, ".json")
    
    table = []
    for i, text in enumerate(x):
        table.append([x[i]] + x_scored[i])

    table = sorted(table, key = lambda l:l[-2], reverse = True)
    table = [{"text": row[0], "score": row[1], "terms_found": row[2]} for row in table]

    with open(fname, "w") as file:
        file.write(json.dumps(table, indent = 4))


def weighted_score_calculation(text, matrix):
    """
    Returns the weighted score for a string (text) based on a matrix of terms and associated
    weights (matrix).

    text - a string for which to calculate the weighted score.
    matrix - a 2D list of each [term, weight] from which to score.
    
    Returns each score and a list of terms found.
    """
    
    score = 0
    included_terms = []
    
    # Generate scores
    for item in matrix:
        value = text.lower().count(item[0].lower()) * float(item[1])
        score += value
        if value > 0:
            included_terms.append(item[0])
        
    return [score, included_terms]