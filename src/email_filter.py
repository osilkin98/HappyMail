# initial commit
# Gmail API imports
import scraper
from scraper import get_gmail_service
from classifier import EmailClassifierModel
import json
from configuration_files import keys
from time import sleep

""" 
    We need to write a function that fetches a list of emails and keep scrolling through
    until we find an email we've read 
"""


# This will just return a list of emails that we can then process
# In the future we should just continue making requests and extending the list until the last_message_id is reached
def get_email_list(service=get_gmail_service(), last_message_id=None):
    """

    :param service service: Google API Service Object, if one is not provided, it'll be automatically generated from\
    scraper.get_gmail_service()
    :param str last_message_id: The ID hash of the message that we recorded last, so we can prevent the program from \
    making too many requests
    :return: A list of Un-decoded Messages in JSON format, as well as the message ID for the first message received,\
     respectively
    :rtype: list
    """
    messages_meta = service.users().messages().list(userId=keys.user_id).execute()

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


def classify_message(message):
    """ Classifies whether a Gmail message expresses positive or negative sentiment

    :param dict message: Gmail message given in JSON
    :return: A value between 1 and 0 of how positive or negative, respectively, the message is
    :rtype: float
    """

    message_texts = scraper.message_to_texts(message)
    classifier = EmailClassifierModel(model_file='models/trained_net.h5')

    probabilities = classifier.predict(message_texts)

    print(probabilities)

    return 0


def classify_messages():
    messages, first_message = get_email_list()

    for message in messages:
        print(json.dumps(message, indent=4))


if __name__== '__main__':
    classify_messages()
