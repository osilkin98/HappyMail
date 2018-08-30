# HappyMail
![NpmLicense](https://img.shields.io/npm/l/express.svg)
![version](https://img.shields.io/badge/version-0.7-brightgreen.svg)

### What is this?
This is (currently) a Python application which uses the Gmail API 
in order to filter out rejection letters for positions that you've applied to.
This is done under the premise that rejection letters are completely 
unnecessary and achieve nothing but lower the applicant's self-confidence. This is 
not intended to create an echo chamber of sorts but to prevent people from becoming 
discouraged in their job searches, especially when they've just graduated and don't
have any work experience, as is the case for me. 

## Installation


Installation is fairly simple but will definitely be improved.
 
To start, clone the repository into any directory, by running 

`$ git clone https://github.com/osilkin98/happymail`


Then run `$ cd HappyMail/` to enter the directory

Now, all you have to do is run `$ python setup.py build_py`

And `setup.py` will handle any and all packages that you need to install.

You'll then be prompted to enter your Gmail username, and you should do so 
for the account that you plan to run the classifier on.  

At this point you'll need to create a Google app which can be done 
pretty painlessly on [this page](https://developers.google.com/gmail/api/quickstart/python)

You'll want to save the file in `HappyMail/src/configuration_files/` as `credentials.json`
as you'll be prompted by the program to do so. Once you've done that, just hit enter and
you'll have another browser tab opened so you can authorize the application to your account.

The necessary subdirectories and files will be installed within the HappyMail directory.
All the data which is downloaded and used is stored underneath the `HappyMail/cache` directory.


## Using Custom Data to Train The Classifier

To Create your own trained classifier, all you want to do is go ahead and create `positive` and 
`negative` labels within your Gmail account. 

Once you have the labels, simply go through your inbox and select what emails you want to see and 
which ones you don't. Once you think you have a good enough size, you just create an instance of 
`EmailClassifier` somewhere in your code, and specify where you'd like the `model_file` to be saved. 

Simply doing something along the lines of this works:
```python
from src.classifier import EmailClassifier

# Instantiates the classifier and handles the 
# Process of pre-processing the data and getting it ready
classifier = EmailClassifier(model_file='path/to/file', data_file='data/file/path')
```

Once you've specified where you want to save the model, simply run the training method
```python
classifier.train_model_with_data()
```

Prediction is as easy as this!
```python
# Outputs the smallest probability value 
classifier.predict(["Is this email something you want to see?",
                    "Or is this something you'd rather see instead?"])

```




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