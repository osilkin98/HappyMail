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
def get_email_list(service=get_gmail_service(), last_message_id=None, max_lookback=None):
    """

    :param Resource service: Google API Service Object, if one is not provided, it'll be automatically generated from\
     scraper.get_gmail_service()
    :param str | None last_message_id: The ID hash of the message that we recorded last, so we can prevent the program from \
     making too many requests
    :param int | None max_lookback: Maximum number of messages to look back through. Leave None for Google default.
    :return: A list of Un-decoded Messages in JSON format, as well as the message ID for the first message received,\
     respectively
    :rtype: list, str
    """

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
            else EmailClassifier(model_file='models/trained_net.h5', auto_train=True)

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


def classify_messages(positive_label, negative_label, threshold=0.1, service=None, max_messages=None):
    """
    Classifies messages within the user's inbox

    :param str positive_label: The ID of the positive label into which we put positive emails
    :param str negative label: The ID of the negative laebl into which we put negative emails
    :param float threshold: Number between 1 and 0, the percentage from 100% into which we will group the emails.\
     Note: it cannot be more than 0.5. If it is greater than 0.5 then 0.5 will be used for the cutoff. Meaning that\
     if it is 51% sure that the message is positive, then it'll be classed as so, but if it is 49% positive,
     the message will be classed as negative.
    :param Resource | None service: Resource object to be used with the Gmail services. If None, it will use
     the scraper.get_gmail_service() function
    :param int | None max_messages: Maximum number of messages to grab
    :return: Nothing
    """
    # Initialize the service if it was not provided to us
    service = service if service is not None else get_gmail_service()

    threshold = threshold if threshold <= 0.5 else 0.5

    # Gets the message list and the first message ID so we know what
    messages, first_message = get_email_list(service=service, max_lookback=max_messages)

    # Email classifier object
    classifier = EmailClassifier(model_file='models/trained_net.h5')

    for message in messages:
        startup = time()
        prob = classify_message(message, classifier)
        end = time()

        # print("Classifying message with ID [{}] took {} secs".format(message['id'], end - startup))

        # print("Message is {:.2%} likely to be negative".format(1-prob))

        # If p >= ~0.9 or whatever 1.0 - threshold gives
        if prob >= 1.0 - threshold:

            # Message is positive
            print(Fore.LIGHTGREEN_EX + "Message was determined to be " + Fore.GREEN + "positive" +
                  Fore.LIGHTGREEN_EX + " with a probability of " + Fore.CYAN + "{:.2%} ".format(prob)+
                  Fore.RESET)





        if prob <= 0.5:
            print("\n\n{} MESSAGE {} IS NEGATIVE {}\n\n".format('#'*15, message['id'], '#'*15))
            print_message(message)


if __name__== '__main__':
    '''
    service = get_gmail_service()

    testing_labels = ('positive_test', 'negative_test')

    labels = scraper.get_specified_labels(testing_labels, service)

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
