from src.classifier import EmailClassifier
from src.email_filter import classify_messages, get_email_list
from src.scraper import get_gmail_service
import src.configuration_files.keys as keys
from time import sleep


def run_main_function(sleep_time=60, max_emails=200):
    """ This function checks to see if a new message has arrived every (sleep_time) seconds, and if it has, it gets
    classified.

    :param int sleep_time: Number of seconds to sleep for before checking again
    :param int max_emails: Maximum number of emails to look back for
    :return: Nothing
    """

    classifier = EmailClassifier(model_file=keys.models + 'trained_net.h5')