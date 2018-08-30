import apiclient
from apiclient.discovery import build
from httplib2 import Http
from oauth2client import file, client, tools
from oauth2client.clientsecrets import InvalidClientSecretsError
from webbrowser import open_new_tab, Error
import os
import base64
import configuration_files.keys as keys
import unicodedata
import json
# import numpy as np
import bs4 as bs
import random
from time import clock


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


def retrieve_credentials(filepath="{}/configuration_files/credentials.json".format(os.getcwd()), retry=3):
    """ Attempts to have the user download the Google API Credentials file in json format

    :param str filepath: Path to API Credentials JSON file
    :param int retry: Number of times to try downloading the credentials file
    :raises: FileNotFoundError if the file was downloaded but not to the correct location, or if it failed to download
    """
    if not os.path.exists(filepath):
        print("could not find file: {}\nRedirecting to Google API key download".format(filepath))

        for i in range(retry):
            open_new_tab("https://console.developers.google.com/apis/credentials?project=email-filter-212723")
            input("Save the credentials file to {}, and press enter when done. ".format(filepath))

            if os.path.exists(filepath):
                break

        # If the filepath was still not found
        if not os.path.exists(filepath):
            print("Failed to save credentials file, raising File not Found error.")
            raise FileNotFoundError("Failed to download {} file passed in as filepath argument.".format(filepath))


# Create Gmail Service
def get_gmail_service(filepath="{}/configuration_files/credentials.json".format(os.getcwd()), scope_mode='modify'):
    """
    :param str filepath: Filepath to Gmail API credentials JSON file
    :param str scope_mode: Scope to use when creating Gmail API Token.
    :return: Gmail API Resource object
    :rtype: Resource | None


    """
    # set the Gmail Scope
    SCOPES = "https://www.googleapis.com/auth/gmail.{}".format(scope_mode)

    store = file.Storage('{}/configuration_files/token.json'.format(os.getcwd()))
    # Try and get the credentials
    creds = store.get()
    flags = tools.argparser.parse_args(args=[])

    # If we failed to get token.json (because we don't have the file)
    if not creds:
        # Create flow object using the credentials json file
        try:
            flow = client.flow_from_clientsecrets(filename=filepath, scope=SCOPES)

        except InvalidClientSecretsError as icse:
            print("{} was caught, likely due to {} not exisitng, trying to download...".format(icse, filepath))

            try:
                retrieve_credentials(filepath=filepath)

            # If for some reason we failed to obtain the credentials
            except FileNotFoundError as fnf:
                print(fnf)
                return None

            flow = client.flow_from_clientsecrets(filename=filepath, scope=SCOPES)

            # Try to download the file and catch the notfound exception

        # Run the flow using our created flow and the Storage object
        creds = tools.run_flow(flow=flow, storage=store, flags=flags)

    # Get the Gmail service
    service = None
    try:
        # Creates the service
        service = build('gmail', 'v1', http=creds.authorize(Http()))

    except apiclient.discovery.HttpError as e:
        print("HttpError: {}\nErrorContents: {}".format(e, e.content))
    except Exception as e:
        print("Standard Exception caused by: {}\n Traceback: {}".format(e.__cause__, e.__traceback__))
    finally:
        # Return the created gmail service
        return service


def message_to_texts(message):
    """ Takes a given message as a JSON element retrieved from the Google API

    :param dict message: A Message Object returned from the Gmail service.users().messages().get() method
    :return: List of texts decoded from base64 into normal texts. This is returned as a list since messages can be\
     fragmented
    :rtype: list
    """
    messages = []

    # If the message is fragmented into parts
    if 'parts' in message['payload']:
        for part in message['payload']['parts']:

            # If we don't have a data field then there's nothing to append here
            if 'data' not in part['body']:
                continue

            soup = bs.BeautifulSoup(base64.urlsafe_b64decode(part['body']['data']).decode('utf-8'),
                                    "html.parser")

            for script in soup(['script', 'style']):
                script.decompose()

            messages.append(soup.get_text())

    else:

        soup = bs.BeautifulSoup(base64.urlsafe_b64decode(
            message['payload']['body']['data']).decode("utf-8"), "html.parser")

        for script in soup(['script', 'style']):
            script.decompose()

        messages.append(soup.get_text())

    return messages


def message_to_texts_traversal(message):
    """ Retrieves data members from within the message by traversing it pre-fix with a stack

    :param dict message: A Message Object returned from the Gmail service.users().messages().get() method
    :return: A list of text fields obtained from within the message. decoded from base64 to plaintext
    :rtype: list
    """

    """ The messages need to be traversed in the following order:
        {
            "payload": { 
                "body": { 
                    "size": <bytesize>
                    // data field may not be present, and so we need to continue looking into parts
                },
                ...,
                // parts contains a list of separate payload objects, each of which needs to be 
                // recursively traversed in pre-fix order, and once we find a body field with
                // a data field inside it, we will decode it from base64 and add it to our list
                // of text items. Then once we've reached the bottom, we will just go back
                // and continue recursing into the next payload object and so forth until we reach the end
                "parts": [ 
                    {
                        "body": { ... },
                        "parts": [ ... ],
                        ...
                    },
                    ...
                ]
            }
        }
        
    the message is structured like a tree where the payload is node that has either 
    one or two key data members. Either it will have a parts child and a body child which 
    means that the message itself is fragmented, or it will have just the body child, 
    which will typically contain data. To ensure that we grab all the data, we traverse it 
    using postfix traversal, in which we will recursively add payloads to the stack and then 
    once we no longer have a parts child we will start looking at the body children and then
    we pop the payload object off the stack and go onto the next one until the stack is 
    empty, then we just return the list of messages
        
        
    """
    # Append the base message payload to the stack
    payload_stack, texts = [message['payload']], []

    while len(payload_stack) > 0:

        current_payload = payload_stack.pop()

        # retrieve the data from the body
        if current_payload['body']['size'] != 0:
            # we have to load the text in as a base64 decoded string of text since it's formatted as HTML
            soup = bs.BeautifulSoup(base64.urlsafe_b64decode(current_payload['body']['data']).decode('utf-8'),
                                    'html.parser')

            # We have to remove all the <script> and <style> elements that don't get removed with get_text()
            for script in soup(['script', 'style']):
                # Delete them
                script.decompose()

            texts.append(soup.get_text())

        if 'parts' in current_payload:
            # We can try iteratively doing this

            for payload in current_payload['parts']:
                payload_stack.append(payload)

            # Or we can try to do this by simple addition
            # payload_stack += current_payload['parts']

    return texts


def get_messages_from_labels(labels, service=get_gmail_service(), include_spam=False):
    """Obtains Messages from the user's defined email labels

    :param dict labels: Dictionary Where the key is the label's name and the value is the label's ID
    :param Resource service: Gmail Resource object which lets us communicate to the user's Gmail account
    :param bool include_spam: Flag to indicate whether or not we should aggregate spam mail alongside our labels
    :return: Returns a list of messages obtained from the labels, and a list of labels at their respective index
    :rtype: list, list
    """
    # Since we want to separate the data from the labels, we'll create
    # Two parallel arrays for the data we retrieve from the Gmail API
    messages, message_labels, message_list = [], [], []

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

            # To write the message meta to the meta messages cache
            with open("{}/meta_messages{:0>5).json".format(keys.list_cache, int(clock()*10000)), "w") as outfile:
                json.dump(messages_meta, fp=outfile, ensure_ascii=False, indent=2)

            # print(messages_meta)

            message_list = []

            # We want to extract the contents of the messages so we have to actually iterate through the
            # list and call the messages().get() method for each message
            for message_meta in messages_meta['messages']:

                # print(message_meta)
                # This returns the full message
                message_full = service.users().messages().get(id=message_meta['id'],
                                                              userId=keys.user_id).execute()

                message_list += message_full

                # print(json.dumps(message_full, indent=2))
                # We add the body of the message to our messages array, and its respective label

                # some of these messages will be segmented in parts so we split up into parts
                # so we just iterate through and extract as much data as we can
                # We might have to implement some form of traversal in order for this to be robust and extensible

                message_texts = message_to_texts(message_full)

                messages += message_texts

                extra_labels = [label] * len(message_texts)

                message_labels += extra_labels

                assert len(message_texts) == len(extra_labels)

                # print("Added texts ({}): {}\nAdded Labels  ({}): {}\n\n".format(len(message_texts), message_texts,
                #                                                                 len(extra_labels), extra_labels))

                # print("Messages: {}\nMessage_labels: {}\n".format(messages, message_labels))

    except apiclient.discovery.HttpError as he:
        print("Got HttpError in get_messages_from_label: {}".format(he))
        print("This is most likely caused by sending too many requests to the Gmail Service")

    except Exception as e:
        print(e)

    finally:
        print("\n\n\n{}{}{}\nMessages ({}): {}\nLabels ({}): {}".format('*'*20, ' TOTAL MESSAGES ', '*'*20,
                                                                        len(messages), messages,
                                                                        len(message_labels), message_labels))

        # Write the files to the the message cache directory
        with open("{}/ScraperMessage{:0>5}.json".format(keys.message_cache, int(clock()*10000)), 'w') as outfile:
            for message in message_list:
                json.dump(message, outfile, ensure_ascii=False, indent=2)
                outfile.write('\n')  # write a newline

        assert len(messages) == len(message_labels)

        return messages, message_labels


# Takes the users labels as input and returns their IDs in a dict
# labels is an iterable array/tuple that contains the names of the desired labels
# Capitalization is required
def get_label_id_dict(labels, service=get_gmail_service()):
    # all_labels is a json object of the form { "labels": [ ... ] }
    all_labels = service.users().labels().list(userId=keys.user_id).execute()

    # To write the labels to the label cache
    with open("{}/labels.json".format(keys.label_cache), 'w') as outfile:
        json.dump(all_labels, fp=outfile, ensure_ascii=False, indent=2)
        outfile.write('\n')  # Write a newline

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


# Read the datafile and use it to extract the training data
# If numeric_labels is set to true, we'll use 1's in place of positive, and 0 for negative
def get_data_from_file(infile="{}/training_data.txt".format(os.getcwd()), numeric_labels=True,
                       create_if_not_found=True, shuffle=True, suppress_warning=False):

    # If the file doesn't exist
    if not os.path.exists(path=infile) and create_if_not_found:
        messages, labels = create_training_data_from_labels()

    # If the file doesn't exist and the user doesn't want to create a new file
    elif not os.path.exists(path=infile) and not create_if_not_found:
        if not suppress_warning:  # Default: throw an exception
            raise FileNotFoundError("The Training Data File wasn't found")
        else:
            return None, None   # If the user chose to suppress warnings

    # Otherwise we can just simply obtain them by using BeautifulSoup
    else:

        datafile = open(file=infile)
        message_tags = bs.BeautifulSoup(datafile, "html.parser").find_all(name="pre", text=True)
        datafile.close()

        messages, labels = [], []

        # iterate through the messages retrieved
        for message in message_tags:
            # append the messages within the <pre> tag to the array and encode them down into utf-8
            messages.append(str(message.contents[0]))

            # do the same for the labels
            labels.append(message["label"].encode("utf-8").decode("utf-8"))
            # break

    # Control mechanism to allow users to retrieve the data without shuffling it
    if shuffle:
        messages, labels = shuffle_messages(messages=messages, labels=labels)

    if numeric_labels:
        # Replace "positive" and "negative" labels with 1s and 0s respectively
        labels = list(map(lambda data: 1 if data == "positive" else 0, labels))

    return messages, labels


# Create Gmail labels
def create_label(name, service=None, userId=keys.user_id,
                 labelListVisibiliy="labelShow", messageListVisibility="show"):
    """ Creates a label using the users.labels,create() method from within the Gmail API

    :param str name: The Name of the Gmail label
    :param Resource | None service: The Gmail Resource Object
    :param str userId: The email address of the user for whom we wish to create the label
    :param str labelListVisibiliy: The visibility of the label in the label list in the Gmail web interface.

     - "labelHide": Do not show the label in the label list,

     - "labelShow": Show the label in the label list. (Default)

     - "labelShowIfUnread": Show the label if there are any unread messages with that label.

    :param messageListVisibility: The visibility of messages with this label in the message list in the
     Gmail web interface.

     - "hide": Do not show the label in the message list.

     - "show": Show the label in the message list. (Default)

    :return: Users.labels Resource of the newly created label
    :rtype: dict
    """

