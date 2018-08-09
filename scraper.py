import apiclient
from apiclient.discovery import build
from httplib2 import Http
from oauth2client import file, client, tools
import os
import base64
import json
# import numpy as np
import keys
import bs4 as bs
import random


# Shuffle the given messages along with their labels
# Assumption is that messages and labels have matching indices
def shuffle_messages(messages, labels, seed=None):
    # We seed the random function so as to get unique swappings each time
    random.seed(a=seed)

    try:
        # Declare this beforehand so as to avoid excess function calls on the stack
        messages_length = len(messages)

        # Since the length of the labels array will only be used in this assert statement,
        # There's no sensible reason to save it
        assert messages_length == len(labels)

        assert messages_length > 1

        k = 0

        for i in range(messages_length):
            # Get a random integer that isn't equal to the current index
            while k == i:
                # Keep generating the random integer
                k = random.randint(a=0, b=messages_length - 1)

            # Swap the messages
            messages[i], messages[k] = messages[k], messages[i]

            # Swap the labels
            labels[i], labels[k] = labels[k], labels[i]

    # in the case that one of the assert statements fail
    except AssertionError as ae:
        print("The messages and label arrays have mismatching lengths, or are less than or equal to 1" +
              "\nMessages length: {}\nLabels length: {}".format(len(messages), len(labels)))

        print("Error printout: {}".format(ae))

    finally:

        return messages, labels


# Create Gmail Service
def get_gmail_service(filepath="{}/credentials.json".format(os.getcwd()), scope_mode='modify'):
    # set the Gmail Scope
    SCOPES = "https://www.googleapis.com/auth/gmail.{}".format(scope_mode)

    store = file.Storage('{}/token.json'.format(os.getcwd()))
    # Try and get the credentials
    creds = store.get()
    flags = tools.argparser.parse_args(args=[])

    # If we failed to get token.json (because we don't have the file)
    if not creds:
        # Create flow object using the credentials json file
        flow = client.flow_from_clientsecrets(filename=filepath, scope=SCOPES)
        # Run the flow using our created flow and the Storage object
        creds = tools.run_flow(flow=flow, storage=store, flags=flags)

    # Get the Gmail service
    service = None
    try:
        # Creates the service
        service = build('gmail', 'v1', http=creds.authorize(Http()))

    except apiclient.discovery.HttpError as e:
        print("HttpError: {}\nErrorContents: {}".format(e.error_details, e.content))
    except Exception as e:
        print("Standard Exception caused by: {}\n Traceback: {}".format(e.__cause__, e.__traceback__))
    finally:
        # Return the created gmail service
        return service


# Args:     Labels is a dict object in which the key is the label's name and the value is the ID
#           Query is a string object, service is (obviously) the Gmail service object, which
#           the default parameter will retrieve should the user fail to provide it
#
# Return:   Returns a list of messages
def get_messages_from_labels(labels, service=get_gmail_service(), include_spam=False):
    # Since we want to separate the data from the labels, we'll create
    # Two parallel arrays for the data we retrieve from the Gmail API
    messages = []
    message_labels = []

    try:

        for label, label_id in labels.items():

            """ returns a json object of the form:
            {
                "messages": [
                    {
                        "id": "message_id"
                        "threadId": "thread_id"
                        "labelIds": [
                            "label_ids"
                            ...
                        ]
                    } 
                    ....
                ]
            }
            
            The actual contents of the message aren't provided, only the IDs so you can make requests to
            retrieve the actual message that the id maps to
            
            """
            label_list = [label_id]

            assert len(label_list) == 1

            # print("Label: {}\nLabel ID: {}\nlabel_list: {}\n\n".format(label, label_id, label_list))
            # Although the labelIds parameter accepts a list of label IDs, we are aggregating by
            # Specific label so we will simply wrap the label_id extracted from the labels
            # Dictionary passed in in a list() call, so it only returns messages associated with one LabelId
            messages_meta = service.users().messages().list(userId=keys.user_id, labelIds=label_list,
                                                            includeSpamTrash=include_spam).execute()

            # print(messages_meta)

            # We want to extract the contents of the messages so we have to actually iterate through the
            # list and call the messages().get() method for each message
            for message_meta in messages_meta['messages']:

                # print(message_meta)
                # This returns the full message
                message_full = service.users().messages().get(id=message_meta['id'],
                                                              userId=keys.user_id).execute()

                # print(json.dumps(message_full, indent=4))
                # We add the body of the message to our messages array, and its respective label

                # some of these messages will be segmented in parts so we split up into parts
                # so we just iterate through and extract as much data as we can
                # We might have to implement some form of traversal in order for this to be robust and extensible
                if 'parts' in message_full['payload']:
                    for part in message_full['payload']['parts']:

                        # if we don't have any data field
                        if 'data' not in part['body']:
                            continue

                        soup = bs.BeautifulSoup(base64.urlsafe_b64decode(part['body']['data']).decode("utf-8"))

                        for script in soup(['script', 'style']):
                            script.decompose()

                        messages.append(soup.get_text())

                        '''
                        # If the message was decoded with single apostrophes
                        if part['body']['data'][-1] == "'":
                            messages.append(base64.urlsafe_b64decode(part['body']['data']).rstrip("'").lstrip("b'"))
    
                        # Otherwise if it was decoded with double apostrophes
                        else:
                            messages.append(base64.urlsafe_b64decode(part['body']['data']).rstrip('"').lstrip('b"'))
    
                        print("Appending (fragmented)")
                        '''
                        message_labels.append(label)

                # Otherwise if the message is whole
                else:

                    soup = bs.BeautifulSoup(base64.urlsafe_b64decode(
                        message_full['payload']['body']['data']).decode("utf-8"))

                    for script in soup(['script', 'style']):
                        script.decompose()

                    messages.append(soup.get_text())

                    message_labels.append(label)
                    '''
                    # Again, if the message was decoded with encapsulating single apostrophe
                    if message_full['payload']['body']['data'][-1] == "'":
                        messages.append(base64.urlsafe_b64decode(
                            message_full['payload']['body']['data']).rstrip("'").lstrip("b'"))
    
                    # otherwise if it was decoded with double apostrophes
                    else:
                        decoded64 = base64.urlsafe_b64decode(message_full['payload']['body']['data'])
                        messages.append(decoded64.rstrip('"').lstrip('b"'))
    
                    print("Appending")'''


                # print("Messages: {}\nMessage_labels: {}\n".format(messages, message_labels))

    except apiclient.errors.HttpError as he:
        print("Got HttpError in get_messages_from_label: {}".format(he))
        print("This is most likely caused by sending too many requests to the Gmail Service")

    except apiclient.errors.Error as e:
        print(e)

    finally:
        return messages, message_labels


# Takes the users labels as input and returns their IDs in a dict
# labels is an iterable array/tuple that contains the names of the desired labels
# Capitalization is required
def get_label_id_dict(labels, service=get_gmail_service()):
    # all_labels is a json object of the form { "labels": [ ... ] }
    all_labels = service.users().labels().list(userId=keys.user_id).execute()

    # This is the actual dict that will be returned
    labels_dict = dict()

    # iterate through all the different label objects that get returned
    for label in all_labels['labels']:

        # If the label's name matches that in the dictionary
        if label['name'] in labels:
            # set the label's ID in our dictionary
            labels_dict[label['name']] = label['id']

    return labels_dict


# Scrape the inbox labels for emails and save them in memory + (write them to a data file)
# None selects the default labels to be used
# The data that gets written to training_data.txt is encoded as base64 to save space
def create_training_data_from_labels(service=get_gmail_service(), outfile=None, overwrite_file=False, labels=None):
    # if the user select the default outfile
    if outfile is None:

        # we simply set it here
        outfile = 'training_data.txt'

        # then we check to see whether or not the file exists, and if we don't want to overwrite it
        if os.path.exists(path='{}/{}'.format(os.getcwd(), outfile)) and not overwrite_file:

            print("The file exists at {}/{} and we chose to not overwrite the file.".format(os.getcwd(), outfile))

            # Then we return
            return None

        else:
            # We concatenate the cwd with the default outfile we declared earlier
            # As it will be
            outfile = "{}/{}".format(os.getcwd(), outfile)

    else:

        if os.path.exists(path='{}'.format(outfile)) and not overwrite_file:
            print("The file exists at {} and we chose not to overwrite the file".format(outfile))

            # Return
            return None

    # if the user selected the default labels
    if labels is None:
        # default labels are positive & negative
        labels = ('positive', 'negative')

    # This returns us a dictionary with the label names as keys and their ID as the value they map to
    labels_dict = get_label_id_dict(labels=labels, service=service)

    messages, message_labels = get_messages_from_labels(labels=labels_dict, service=service, include_spam=True)

    try:
        # try and open the outfile if it already
        mode_key = 'x' if not overwrite_file else 'w'  # type: str
        with open(file=outfile, mode=mode_key) as datafile:

            for i in range(len(messages)):
                datafile.write("<pre label=\"{}\">\n{}\n</pre>\n".format(message_labels[i], messages[i]))

        print("Successfully wrote data to {}".format(outfile))

    # If we put it in the mode to not overwrite the file and it throws the error then we catch it and print it out
    except FileExistsError as fee:
        print(fee)

    finally:
        return messages, message_labels


# Read the file and
def get_data_from_file(infile="{}/training_data.txt".format(os.getcwd()), create_if_not_found=True):

    # If the file doesn't exist
    if not os.path.exists(path=infile) and create_if_not_found:
        messages, labels = create_training_data_from_labels()

    # Otherwise we can just simply obtain them by using BeautifulSoup
    else:

        datafile = open(file=infile)
        message_tags = bs.BeautifulSoup(datafile, "html.parser").find_all(name="pre")
        datafile.close()

        messages, labels = [], []

        for message in message_tags:
            messages.append(message.contents[0].decode("utf-8"))
            labels.append(message["label"].decode("utf-8"))

    return shuffle_messages(messages=messages, labels=labels)
