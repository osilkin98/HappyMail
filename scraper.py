import apiclient
from apiclient.discovery import build
from httplib2 import Http
from oauth2client import file, client, tools
import os
import numpy as np
import keys


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
            # Although the labelIds parameter accepts a list of label IDs, we are aggregating by
            # Specific label so we will simply wrap the label_id extracted from the labels
            # Dictionary passed in in a list() call, so it only returns messages associated with one LabelId
            messages_meta = service.users().messages().list(userId=keys.user_id, labelIds=list(label_id),
                                                            includeSpamTrash=include_spam).execute()

            # We want to extract the contents of the messages so we have to actually iterate through the
            # list and call the messages().get() method for each message
            for message_meta in messages_meta['messages']:

                # This returns the full message
                message_full = service.users().messages().get(id=message_meta['id'],
                                                              userId = keys.user_id).execute()

                # We add the body of the message to our messages array, and its respective label
                messages.append(message_full['body'])
                message_labels.append(label)

    except apiclient.errors.HttpError as he:
        print(he)

    except apiclient.errors.Error as e:
        print(e)

    except Exception as e:
        print(e)

    finally:
        return messages, message_labels


# Scrape the inbox labels for emails and save them in memory + (write them to a data file)
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

        if os.path.exists(path='{}'.format(outfile)) and not overwrite_file:

            print("The file exists at {} and we chose not to overwrite the file".format(outfile))

            # Return
            return None

    # if the user selected the default labels
    if labels is None:

        # default labels are positive & negative
        labels = {'positive': '', 'negative': ''}



    # all_labels is a json object of the form { "labels": [ ... ] }
    all_labels = service.users().labels().list(userId=keys.user_id).execute()

    # iterate through all the different label objects that get returned
    for label in all_labels['labels']:

        # If the label's name matches that in the dictionary
        if label['name'] in labels:

            # set the label's ID in our dictionary
            labels[label['name']] = label['id']

            # we could perhaps write to a file at this point since we have

            # TODO: implement function to go through and get the information from the labels into the file

    return labels