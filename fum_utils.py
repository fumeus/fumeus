import bs4, chardet, nltk, os


def check_extension(fname, extension = ".csv"):
    """
    Checks whether the fname includes an extension.
    Adds an extension if none exists.

    fname - the name of the file to check.
    extension - the extension to append if necessary.
     >> Default: ".csv".
    """

    root, ending = os.path.splitext(fname)
    if not ending:
        ending = extension
    return root + ending


def clean_documents(x, stop_words = []):
    """
    Removes HTML, stopwords, and words of fewer than three characters from text.
    x - a list of textual documents (strings).
    
    Returns a list of cleaned textual documents (strings).
    """
    
    return tuple(tuple(word for word in \
            nltk.RegexpTokenizer("[a-z]{3,}").tokenize(bs4.BeautifulSoup(document.lower(),
            features = "html.parser").get_text()) if word not in stop_words) for document in x)


def guess_encoding(fname):
    """
    Guesses the encoding of a file.
    fname - the name of the file whose encoding to guess.

    Returns the guessed encoding of the file (string).
    """

    with open(fname, "rb") as file:
        result = chardet.detect(file.read())
    return result["encoding"]