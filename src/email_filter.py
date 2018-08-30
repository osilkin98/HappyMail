# initial commit
# Gmail API imports
import scraper
from scraper import get_gmail_service
from classifier import EmailClassifier
import json
from configuration_files import keys
from time import sleep, time, clock
from colorama import Fore

""" 
    We need to write a function that fetches a list of emails and keep scrolling through
    until we find an email we've read 
"""


def print_message(message, snippet=True):
    """Prints a Gmail Message to the terminal

    :param dict message: Gmail message formatted from JSON format into a Python dict
    :param bool snippet: Flag to indicate whether we should print a snippet of the message or the whole thing. \
     On by default.
    :return: Nothing
    """

    if snippet:
        print("{}\nMESSAGE ID: {}\nSnippet: {}\n{}".format('#'*len(message['snippet']),
                                                           message['id'], message['snippet'],
                                                           '#'*len(message['snippet'])))
    else:
        # To obtain the actual message text as a list
        message_parts = scraper.message_to_texts_traversal(message)

        max_length = 0

        # To set the maximum length for the part formatting
        for message in message_parts:
            max_length = max(len(message), max_length)

        for message in message_parts:
            print("{}\n{}\n".format('-'*max_length, message))

        print('-'*max_length)


# This will just return a list of emails that we can then process
# In the future we should just continue making requests and extending the list until the last_message_id is reached
def get_email_list(service=None, last_message_id=None, max_lookback=None):
    """ Fetches a list of Gmail emails linked to the account specified in configuration_files.keys.
    The email list is then saved to keys.list_cache/email_list(time).json

    :param Resource service: Google API Service Object, if one is not provided, it'll be automatically generated from\
     scraper.get_gmail_service()
    :param str last_message_id: The ID hash of the message that we recorded last, so we can prevent the program
     from making too many requests
    :param int max_lookback: Maximum number of messages to look back through. 100 if None is specified
    :return: A list of Un-decoded Messages in JSON format, as well as the message ID for the first message received,\
     respectively
    :rtype: list, str
    """

    service = service if service is not None else scraper.get_gmail_service()

    messages_meta = service.users().messages().list(
        userId=keys.user_id).execute() if max_lookback is None else service.users().messages().list(
        userId=keys.user_id, maxResults=abs(max_lookback)).execute()

    # To write the files to the cache
    with open("{}/email_list{:0>5}.json".format(keys.list_cache, int(clock()*10000)), 'w') as outfile:
        json.dump(messages_meta, fp=outfile, ensure_ascii=False, indent=2)
        outfile.write('\n')

    first_message_id = None

    messages = []

    for message_meta in messages_meta['messages']:

        # Set the first message ID header
        if first_message_id is None:
            first_message_id = message_meta['id']

        # if we already processed these emails before
        if last_message_id is not None and last_message_id == message_meta['id']:

            # To write the files to the cache before we exit from the loop
            with open("{}/EmailListMessage{:0>5}.json".format(keys.message_cache, int(clock() * 10000)),
                      'w') as outfile:

                # Dump each message as a json file
                for message in messages:
                    # Dump the json message
                    json.dump(message, outfile, ensure_ascii=False, indent=2)
                    # Write a newline
                    outfile.write('\n')

            return messages, first_message_id

        full_message = service.users().messages().get(id=message_meta['id'], userId=keys.user_id).execute()

        messages.append(full_message)

    # To write the files to the cache
    with open("{}/EmailListMessage{:0>5}.json".format(keys.message_cache, int(clock()*10000)), 'w') as outfile:

        # Go through each message and dump it as a json object
        for message in messages:
            json.dump(message, outfile, ensure_ascii=False, indent=2)

            # Write a newline following the json object
            outfile.write('\n')

    return messages, first_message_id


def classify_message(message, classifier=None):
    """ Classifies whether a Gmail message expresses positive or negative sentiment

    :param dict message: Gmail message given in JSON
    :param EmailClassifier | None classifier: The classifier model which will be used to inspect the emails\
     If None is specified, it'll default to using the one trained in 'models/trained_net.h5'.
     If it doesn't exist, then it will attempt to train itself.
    :return: A value between 1 and 0 of how positive or negative, respectively, the message is.
    :rtype: float
    """
    try:
        classifier = classifier if classifier is not None \
            else EmailClassifier(model_file=keys.models+'/trained_net.h5', auto_train=True)

        message_texts = scraper.message_to_texts_traversal(message)

        assert len(message_texts) > 0

        probabilities = classifier.predict(message_texts)

        print("Probabilities: {}, \nlowest probability attained: {}".format(
            probabilities, min(probabilities)))

        return min(probabilities)[0]

    except AssertionError:
        print("Message with ID '{}' gave back no classifiable text:\n\nJSON Dump:\n{}".format(
            message['id'], json.dumps(message, indent=4)))

        return 1


def classify_messages(negative_label, positive_label=None, messages=None, classifier=None,
                      threshold=0.1, service=None, max_messages=None, auto_train=True,
                      quiet=False, logging=True):
    """
    Classifies messages within the user's inbox. The classified messages are reflected within the user's
    inbox. Additionally, The model also saves logs to the cache/processed/ directories.

    :param str negative_label: The ID of the negative laebl into which we put negative emails.
    :param str positive_label: The ID of the positive label into which we put positive emails. If None is specified,
     the program will simply ignore the positive labels
    :param list messages: List of Gmail message objects, if none is specified then get_email_list is called by default.
    :param EmailClassifier classifier: EmailClassifier model compiled with Keras as a trained model. If None, then one\
     is called and if it has no data then it will self-train, unless otherwise specified.
    :param float threshold: Number between 1 and 0, the percentage from 100% into which we will group the emails.\
     Note: it cannot be more than 0.5. If it is greater than 0.5 then 0.5 will be used for the cutoff. Meaning that\
     if it is 51% sure that the message is positive, then it'll be classed as so, but if it is 49% positive,
     the message will be classed as negative.
    :param Resource service: Resource object to be used with the Gmail services. If None, it will use
     the scraper.get_gmail_service() function
    :param int | None max_messages: Maximum number of messages to grab
    :param bool auto_train: Flag to specify whether or not we should automatically train the model if no data
     was found
    :param bool quiet: Flag for whether or not we should output messages to the console
    :param bool logging: Flag for whether or not we should saves files as json to cache/processed
    :return: Nothing
    """

    threshold = threshold if positive_label is None and threshold <= 1.0 \
        else threshold if positive_label is not None and threshold <= 0.5\
        else 1.0 if positive_label is None and threshold > 1.0 \
        else 0.5  # if the positive label is active and threshold exceeds 0.5

    # Initialize the service if it was not provided to us
    service = service if service is not None else get_gmail_service()

    # If there were no messages passed in
    if messages is None:

        # Gets the message list and the first message ID so we know what
        messages, first_message = get_email_list(service=service, max_lookback=max_messages)

    # Email classifier object
    classifier = classifier if classifier is not None else EmailClassifier(model_file=keys.models + '/trained_net.h5',
                                                                           auto_train=auto_train)

    # Set the bodies here so there's not more time spent on initializing the functions
    negative_body = {'removeLabelIds': [], 'addLabelIds': [negative_label]}  # For negative messages

    # For positive messages
    positive_body = None if positive_label is None else {'removeLabelIds': [],
                                                         'addLabelIds': [positive_label]}
    for message in messages:
        # To time the function
        startup = time()
        prob = classify_message(message, classifier)
        end = time()

        response = None

        # I don't really like the repetitive structure here but it's a binary problem so whatever

        # If the probability is likely to be negative
        if prob <= (0 + threshold):
            if not quiet:
                print(Fore.LIGHTRED_EX + "Messaged was determined to be " + Fore.RED + "negative" +
                      Fore.LIGHTRED_EX + " with a probability of " + Fore.CYAN + '{:.2%} '.format(prob) +
                      Fore.RESET)

            response = service.users().messages().modify(userId=keys.user_id, id=message['id'],
                                                         body=negative_body).execute()
        elif positive_label is not None and prob >= (1.0 - threshold):

            # Message is positive
            if not quiet:
                print(Fore.LIGHTGREEN_EX + "Message was determined to be " + Fore.GREEN + "positive" +
                      Fore.LIGHTGREEN_EX + " with a probability of " + Fore.CYAN + "{:.2%} ".format(prob) +
                      Fore.RESET)

            response = service.users().messages().modify(userId=keys.user_id, id=message['id'],
                                                         body=positive_body).execute()

        if logging and response is not None:
            with open(keys.processed_responses + '/response' + message['id'] + '.json', 'w') as outfile:
                json.dump(response, outfile, ensure_ascii=False, indent=2)

        # Set the new attributes
        message['positive_probability'] = str(prob)
        message['classification_time'] = str(end - startup)

        if logging:
            with open(keys.processed_messages + '/message{}.json'.format(message['id']), 'w') as outfile:
                json.dump(message, outfile, ensure_ascii=False, indent=2)


if __name__== '__main__':
    classify_messages('Label_4', threshold=0.3)
    '''
    service = get_gmail_service()

    testing_labels = ('positive_test', 'negative_test')

    labels = scraper.get_specified_labels(testing_labels, service)
Label_5
    # go through each label in the testing labels list
    for label in testing_labels:

        # If it's not in the label keys dict that we retrieved
        if label not in labels.keys():
            # We create the label we need
            new_label = scraper.create_label(label, service)

            print("Created {}".format(json.dumps(new_label, indent=2)))

    '''

    '''
    # For storing the texts for testing
    texts_normal_method, texts_traversal_method = [], []

    # For timing the actual methods
    total_normal_time, total_traversal_time, normal_time, traversal_time, start, stop = 0, 0, 0, 0, 0, 0

    normal_length, traversal_length = 0, 0

    text = None

    for message in messages:

        start = time()
        text = scraper.message_to_texts(message)
        stop = time()

        texts_normal_method.append(text)

        normal_length = len(text)

        print("{}\n\nNormal method obtained({}): \n{}\n".format(
            '#'*200, normal_length, text))

        normal_time = stop - start

        total_normal_time += normal_time

        start = time()
        text = scraper.message_to_texts_traversal(message)
        stop = time()

        traversal_time = stop - start

        total_traversal_time += traversal_time

        traversal_length = len(text)

        texts_traversal_method.append(text)

        print("{}\n\nTraversal method obtained({}):\n{}\n".format('-'*200, traversal_length, text))

        print("{}\n\nNormal Time: {}s\nTraversal Time: {}s\n".format('_'*200, normal_time, traversal_time))

        print("# of elements obtained w/ normal method: {}\n# of elements obtained w/ traversal method: {}\n".format(
            normal_length, traversal_length
        ))

    print("{}\nNormal Time:\nTotal: {}s\nAverage: {}s\n{}\nTraversal Time:\nTotal: {}s\nAverage: {}s\n\n".format(
        '#'*50, total_normal_time, total_normal_time/len(messages),
        '-'*50, total_traversal_time, total_traversal_time/len(messages)))

    print("Length of total normal texts: {}\nLength of total traversal texts: {}".format(len(texts_traversal_method),
                                                                                         len(texts_traversal_method)))
    '''
