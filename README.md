**Fumeus** is a family of Python tools for performing smoke term analyses. "Smoke terms" were first popularized in safety surveillance as sets of words and phrases particularly indicative of safety hazard-related language. More broadly, these methods can be utilized to develop sets of words and phrases associated with a dependent variable and to rank or score text to surveil for that dependent variable.

This repository contains the source code for Fumeus. You can also use Fumeus through its web interface at [fumeus.app](https://fumeus.app).

**Fumeus** has two core functions:
- *Generate smoke terms*: Upload a CSV file of textual narratives and classifications. Specify an information retrieval approach and an n-gram length, which will be used to parse records and determine potentially predictive n-grams. Access results in CSV or JSON.
- *Score records*: Upload a CSV file of textual narratives. Upload a separate CSV file of a weighted n-gram dictionary. Textual narratives will be scored and ranked by the specified dictionary. Access results in CSV or JSON.
