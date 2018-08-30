# HappyMail
![NpmLicense](https://img.shields.io/npm/l/express.svg)
![version](https://img.shields.io/badge/version-0.7-brightgreen.svg)

This is (currently) a Python application which uses the Gmail API 
in order to filter out rejection letters for positions that you've applied to.
This is done under the premise that rejection letters are completely 
unnecessary and achieve nothing but lower the applicant's self-confidence. This is 
not intended to create an echo chamber of sorts but to prevent people from becoming 
discouraged in their job searches, especially when they've just graduated and don't
have any work experience, as is the case for me. 

## The Technology
The Project uses a one-directional LSTM that uses learned word embeddings to 
determine whether or not an email is negative in sentiment. This will be replaced 
very soon with a bi-directional LSTM so the model can be more accurate, although
it's not entirely necessary as rejection letters typically follow the format
"...We regret to inform you..." or "...we have decided to pursue other candidates..."

This project is in a very primtive stage right now and relies on the Google API, 
and as a result cannot be extended to any domain that does not operate through 
Google services. This will change.

This project currently only operates on the user's host system, and requires 24/7 
desktop computer uptime for usage. This will also change. 