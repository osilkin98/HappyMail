# initial commit
# Gmail API imports
import scraper
from scraper import get_gmail_service
from classifier import EmailClassifier
import json
from configuration_files import keys
from time import sleep
from time import time

""" 
    We need to write a function that fetches a list of emails and keep scrolling through
    until we find an email we've read 
"""


def print_message(message):
    """Prints a Gmail Message to the terminal

    :param dict message: Gmail message formatted from JSON format into a Python dict
    :return: Nothing
    """

    # To obtain the actual message text as a list
    message_parts = scraper.message_to_texts(message)

    max_length = 0

    # To set the maximum length for the part formatting
    for message in message_parts:
        max_length = max(len(message), max_length)

    for message in message_parts:
        print("{}\n{}\n".format('-'*max_length, message))

    print('-'*max_length)


# This will just return a list of emails that we can then process
# In the future we should just continue making requests and extending the list until the last_message_id is reached
def get_email_list(service=get_gmail_service(), last_message_id=None, max_lookback=None):
    """

    :param service service: Google API Service Object, if one is not provided, it'll be automatically generated from\
    scraper.get_gmail_service()
    :param str last_message_id: The ID hash of the message that we recorded last, so we can prevent the program from \
    making too many requests
    :param int | None max_lookback: Maximum number of messages to look back through. Leave None for Google default.
    :return: A list of Un-decoded Messages in JSON format, as well as the message ID for the first message received,\
     respectively
    :rtype: list, str
    """

    messages_meta = service.users().messages().list(
        userId=keys.user_id).execute() if max_lookback is None else service.users().messages().list(
        userId=keys.user_id, maxResults=abs(max_lookback)).execute()

    first_message_id = None

    messages = []

    for message_meta in messages_meta['messages']:

        # Set the first message ID header
        if first_message_id is None:
            first_message_id = message_meta['id']

        # if we already processed these emails before
        if last_message_id is not None and last_message_id == message_meta['id']:
            return messages, first_message_id

        full_message = service.users().messages().get(id=message_meta['id'], userId=keys.user_id).execute()

        messages.append(full_message)

    return messages, first_message_id


def classify_message(message, classifier=EmailClassifier(model_file='models/trained_net.h5', auto_train=True)):
    """ Classifies whether a Gmail message expresses positive or negative sentiment

    :param dict message: Gmail message given in JSON
    :param EmailClassifier classifier: The classifier model which will be used to inspect the emails\
     If None is specified, it'll default to using the one trained in 'models/trained_net.h5'.
     If it doesn't exist, then it will attempt to train itself.
    :return: A value between 1 and 0 of how positive or negative, respectively, the message is, \
     along with a list of all the other probabilities.
    :rtype: float, list
    """

    message_texts = scraper.message_to_texts(message)

    probabilities = classifier.predict(message_texts)

    '''
    max_len = 0
    for text in message_texts:
        max_len = max(len(text), max_len)

    for i in range(len(message_texts)):
        print("{}\n\nMessage Text({}): \"{}\"\nProbability({}): {}\n".format(
            '#' * max_len, i, message_texts[i], i, probabilities[i]))
    '''

    print("Probabilities: {}, \nlowest probability attained: {}".format(
        probabilities, min(probabilities)))

    return min(probabilities)[0], probabilities, classifier

def classify_messages(max_messages=None):
    """

    :param int | None max_messages: Maximum number of messages to grab
    :return: Nothing
    """
    messages, first_message = get_email_list(max_lookback=max_messages)




if __name__== '__main__':
    email_list, first = get_email_list(max_lookback=1)

    print(json.dumps(email_list, indent=4))
    for message in email_list:
        classify_message(message)