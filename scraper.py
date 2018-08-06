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

    # TODO: scrape the gmail api for the data

    # all_labels is a json object of the form { "labels": [ ... ] }
    all_labels = service.users().labels().list(userId=keys.user_id).execute()

    # iterate through all the different label objects that get returned
    for label in all_labels['labels']:

        # If the label's name matches that in the dictionary
        if label['name'] in labels:

            # set the label's ID in our dictionary
            labels[label['name']] = label['id']

            # we could perhaps write to a file at this point since we have