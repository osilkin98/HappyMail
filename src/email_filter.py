# initial commit
# Gmail API imports
from scraper import get_gmail_service
import classifier
from configuration_files import keys

""" 
    We need to write a function that fetches a list of emails and keep scrolling through
    until we find an email we've read 
"""

def get_email_list(service=get_gmail_service(), last_message_id=None):
    """

    :param service: Google API Service Object, if one is not provided, it'll be automatically generated from scraper.get_gmail_service()
    :param last_message_id: The ID hash of the message that we recorded last, so we can prevent the program from making too many requests
    :return:
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