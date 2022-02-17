import csv, json, math, nltk
from fum_utils import *


def build_ngram_matrix(documents, length, y):
    """
    Builds a 2-D list (matrix) the ngrams of a desired length in each document with its final
    classification.
    
    documents - a list of textual documents (strings).
    length - the length of the ngrams to generate (e.g., 2 for bigrams).
    y - a list of the classifications (1 or 0) for each document.
    
    Returns a 2-D list (matrix) where each row contains ((ngrams,), classification).
    """
    
    return tuple((tuple(" ".join(term) for term in tuple(nltk.everygrams(document, 
            min_len = length, max_len = length))), y[i]) for i, document in enumerate(documents))


def ir_score(ngrams, ngram_matrix, y, mode = "CC"):
    """
    Implements the information retrieval algorithms for ranking ngrams based upon their
    prevalence in relevant verus irrelevant documents.
    
    Fan, Weiguo, Michael D. Gordon, and Praveen Pathak.
    "Effective profiling of consumer information retrieval needs:
    A unified framework and empirical comparison."
    Decision Support Systems 40.2 (2005): 213-233.

    ngrams - a list of unique ngrams to analyze.
    ngram_matrix - a 2-D list (matrix) where each row contains ((ngrams,), classification).
    y - a list of the classifications (1 or 0) for each document.
    mode - "CC" for "Correlation Coefficient"; "DRC" for "Document and Relevance Correlation";
    "RSV" for "Robertson's Selection Value"; or "RCV" for "Relevance Correlation Value".
     >> Default: "CC".

    Returns a dictionary of {ngram: score}.
    """
    
    # Initialize IR score container
    ir_scores = {ngram: {"a": 0, "b": 0, "c": 0, "d": 0} for ngram in ngrams}
    
    # For each ngram, update a and b (containing) counters
    for document, relevance in ngram_matrix:
        for ngram in document:
            if relevance == 1:
                ir_scores[ngram]["a"] += 1
            else:
                ir_scores[ngram]["b"] += 1
                
    # Determine number of total ones and zeros
    count_1 = y.count(1)
    count_0 = y.count(0)
    
    # Calculate each IR score
    for ngram in ir_scores:
        
        # Complete contingency table by determining complements of a and b
        ir_scores[ngram]["c"] = count_1 - ir_scores[ngram]["a"]
        ir_scores[ngram]["d"] = count_0 - ir_scores[ngram]["b"]
        
        try:
            if mode == "CC":
                ir_scores[ngram]["score"] = (math.sqrt(ir_scores[ngram]["a"] + ir_scores[ngram]["b"] + \
                        ir_scores[ngram]["c"] + ir_scores[ngram]["d"]) * ((ir_scores[ngram]["a"] * \
                        ir_scores[ngram]["d"]) - (ir_scores[ngram]["c"] * ir_scores[ngram]["b"]))) / \
                        (math.sqrt((ir_scores[ngram]["a"] + ir_scores[ngram]["b"]) * \
                        (ir_scores[ngram]["c"] + ir_scores[ngram]["d"])))
            elif mode == "RSV":
                ir_scores[ngram]["score"] = ir_scores[ngram]["a"] * math.log10((ir_scores[ngram]["a"] * ir_scores[ngram]["d"]) / (ir_scores[ngram]["b"] * ir_scores[ngram]["c"]))
            elif mode == "RCV":
                ir_scores[ngram]["score"] = ir_scores[ngram]["a"] / (math.sqrt(ir_scores[ngram]["a"] + ir_scores[ngram]["b"]) * math.sqrt(ir_scores[ngram]["a"] + ir_scores[ngram]["c"]))
            elif mode == "DRC":
                ir_scores[ngram]["score"] = ir_scores[ngram]["a"] ** 2 / math.sqrt(ir_scores[ngram]["a"] + ir_scores[ngram]["b"]) 

        except (ZeroDivisionError, ValueError):
            ir_scores[ngram]["score"] = 0
            
    return {ngram: ir_scores[ngram]["score"] for ngram in ir_scores}


def csv_to_list(dataset_fname, header = False, y_column = -1):
    """
    Reads a csv file and returns the contents of that file as a 2-D list.
    Delineates x (independent) variables from y (dependent) variable.

    dataset_fname - the file name for the dataset to read.
    header - whether the dataset contains a header row (Boolean).
     >> Default: False.
    y_column - the index of the column to treat as y data.
     >> Default: -1.

    Returns x (2-D list), y (list).
    """

    # Initialize data containers and check extension of provided file name
    x = []
    y = []
    bad_data = {}
    dataset_fname = check_extension(dataset_fname)

    encoding = guess_encoding(dataset_fname)
    with open(dataset_fname, "r", encoding = encoding) as file:
        reader = csv.reader(file, delimiter = ",")
        for i, line in enumerate(reader):
            row = []

            # If negative index is used, find equivalent forward-looking index
            if i == 0 and y_column < 0:
                y_column = len(line) + y_column

            # Check for header row
            if i > 0 or not header:
                for j, item in enumerate(line):

                    # Collect y values
                    if j == y_column:
                        try:
                            y.append(float(item))
                        except ValueError:
                            y.append(0)
                        except:
                            pass
                        continue

                    # Treat empty cells as zero
                    if len(item) == 0:
                        row.append(0)
                        continue

                    # Attempt to convert each element to float
                    # Track fields that cannot be converted
                    try:
                        row.append(float(item))
                    except ValueError:
                        if j not in bad_data:
                            bad_data[j] = [item]
                        else:
                            bad_data[j].append(item)
                    except:
                        pass

                x.append(row)

    # Check that bad data was present and handle
    if len(bad_data) > 0:
        bad_field_ids = sorted(tuple(bad_data.keys()))
        if len(x) == 0:
            x = [[] * len(bad_data[bad_field_ids[0]])]
        for field_id in bad_field_ids:
            for i, item in enumerate(bad_data[field_id]):
                x[i].append(item)

    return x, y


def get_unique_ngrams(documents, length):
    """
    Given a set of documents and a desired length, generates all possible ngrams of that length.
    
    documents - a list of textual documents (strings).
    length - the length of the ngrams to generate (e.g., 2 for bigrams).
    
    Returns a list of unique ngrams of the specified length in the documents.
    """

    unique_ngrams = set()
    for document in documents:
        for term in nltk.everygrams(document, min_len = length, max_len = length):
            term_str = " ".join(term)
            unique_ngrams.add(term_str)

    return unique_ngrams


def main(dataset_fname, header = False, x_column = 0, y_column = -1, ngram_length = 1, mode = "CC", output_fname = "output.csv", top_n = 200, format = "csv"):
    """
    Main driver code to perform text analysis.
    Loads dataset from a CSV file.
    Performs text cleaning on loaded text data.
    Generates unique n-grams (unigrams, bigrams, and trigrams).
    Creates output CSV files containing results.
    
    dataset_fname - the file name for the dataset to read.
    header - whether the dataset contains a header row (Boolean).
     >> Default: False.
    x_column - the index of the column to treat as x (textual) data.
     >> Default: 0.
    y_column - the index of the column to treat as y data.
     >> Default: -1.
    ngram_length - the length of the ngrams to generate (e.g., 2 for bigrams).
     >> Default: 1.
    mode - "CC" for "Correlation Coefficient"; "DRC" for "Document and Relevance Correlation";
    "RSV" for "Robertson's Selection Value"; or "RCV" for "Relevance Correlation Value".
     >> Default: "CC".
    output_fname - the file name for the output file.
     >> Default: "output.csv".
    top_n - length of each n-gram list to output to CSV files.
     >> Default: 200.
    format - the output format to utilize (either CSV or JSON).
     >> Default: "csv".
    """

    # Load data from a CSV file
    # x is the textual data
    # y is the dependent variable, 1 or 0
    x, y = csv_to_list(dataset_fname, header, y_column)

    # Clean text data
    x_clean = clean_documents([document[x_column] for document in x])
    
    # Generate n-grams and build matrices for each document
    ngrams = get_unique_ngrams(x_clean, ngram_length)
    ngram_matrix = build_ngram_matrix(x_clean, ngram_length, y)

    # Run scoring algorithm
    ngram_scores = ir_score(ngrams, ngram_matrix, y, mode.upper())

    # Output results to CSV or JSON files
    if format == "csv":
        terms_to_csv(ngram_scores, output_fname, top_n)
    elif format == "json":
        terms_to_json(ngram_scores, output_fname, top_n)


def make_terms_table(terms_dict, n = None):
    """
    Given a dictionary of terms and corresponding values, reformats in tabular (2-D list) format.
    
    terms_dict - a dictionary of {term: value}.
    n - optionally, the number of (top) records considered.
     >> Default: None.
     
    Returns a 2-D list of sorted (term, value) in descending order.
    """
    
    # Set n to length of dataset if not specified
    if not n:
        n = len(terms_dict)
    
    return sorted(terms_dict.items(), key = lambda x: x[1], reverse = True)[:n]


def terms_to_csv(terms_dict, fname, n = None):
    """
    Writes a list of terms to an output CSV file.
    
    terms_dict - a dictionary of {term: value}.
    fname - the file name for the output file.
    n - optionally, the maximum number of terms to include.
     >> Default: None.
    """
        
    terms_table = make_terms_table(terms_dict, n)
    
    fname = check_extension(fname)
    with open(fname, "w") as file:
        writer = csv.writer(file, lineterminator = "\n")
        for row in terms_table:
            writer.writerow(row)


def terms_to_json(terms_dict, fname, n = None):
    """
    Writes a list of terms to an output JSON file.
    
    terms_dict - a dictionary of {term: value}.
    fname - the file name for the output file.
    n - optionally, the maximum number of terms to include.
     >> Default: None.
    """
        
    terms_table = make_terms_table(terms_dict, n)
    terms_table = [{"term": row[0], "score": row[1]} for row in terms_table]

    fname = check_extension(fname, ".json")

    with open(fname, "w") as file:
        file.write(json.dumps(terms_table, indent = 4))