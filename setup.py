from subprocess import call
from sys import executable
from distutils.core import setup
from distutils.command.build_py import build_py
import os
from os.path import exists
from getpass import getuser


try:
    import colorama
except ImportError as IE:
    print("Module Colorama not found, installing")

    call([executable, '-m', 'pip', 'install','--user', 'colorama==0.3.9'])

finally:
    from colorama import Fore


needed_packages = ['apiclient>=1.0.3',
                   'httplib2>=0.9.2',
                   'google-api-python-client-py3>=1.2',
                   'oauth2client>=4.1.2',
                   'bs4>=0.0.1',
                   'tensorflow>=1.10.1' if call(['which', 'nvidia-smi']) != 0 else 'tensorflow-gpu>=1.10.1',
                   'keras>=2.2.2']

needed_directories = {"config_files": "src/configuration_files",
                      "models": "models",
                      "logdir": "models/logs",
                      "cache_dir": "cache",
                      "message_cache": "cache/messages",
                      "label_cache": "cache/labels",
                      "list_cache": "cache/lists"}


def install_packages(packages):
    """
    :param list packages: List of Python Package names to be installed by pip in the format 'package-name>=version.number'
    :return: Nothing
    """
    for package in packages:
        # print("installing package {} with pip" .format(package))
        pip_command = "{} -m pip install {} --user".format(executable, package)
        # print("Running {}".format(pip_command))
        retcode = call(pip_command.split(' '))

        if retcode is not 0:
            print(Fore.RED + "return code was {} when trying to install {}".format(retcode, packages))
        else:
            print(Fore.GREEN + "installed {}".format(package))

        print(Fore.RESET)


# To create directories
def create_subdirectories(directories, base_dir=os.getcwd()):
    """ this function creates the directories and __init__.py files underneath the given path

    :param list | dict | tuple directories: List of subdirectories within the current directory to create. \
     If it's a a dict then the directories should be the values that the keys get mapped to.
    :param str base_dir: Path to the base directory under which all the directories will be installed.\
     this parameter *should* be an absolute path for universal use.
    :return: A list of the directories but with the base_dir prepended to them
    """

    # Strip the tailing / symbol from the base directory for consistency
    base_dir = base_dir.rstrip('/')

    full_directories = []

    # Try to create the subdirectories
    try:

        # Check to see whether or not we're dealing with a dict, and loop over all the
        # Subdirectories specified in the iterable to create them and populated the current path
        for directory in (directories if type(directories) != dict else directories.values()):

            # Saves the full path to the directory and strips it on the right for consitency
            full_dir = base_dir + '/' + directory.rstrip('/')

            # Save it to the full_directories list
            full_directories.append(full_dir)

            # If the directory doesn't already exist, then we create it
            if not exists(full_dir):

                print(Fore.YELLOW +  full_dir + " doesn't exist")

                # Call the makedirs function to create all the directories in between
                os.makedirs(full_dir)

                print(Fore.GREEN + "Created directory " + full_dir + Fore.RESET)

            # If the __init__.py file doesn't exist
            if not exists(full_dir + "/__init__.py"):

                # Create the __init__.py file within the directories
                with open(full_dir + "/__init__.py", "w") as init:

                    # Note that it wa generated by the current file
                    init.write("# This file was generated by {}\n".format(__file__))

                print(Fore.GREEN + "Created "+full_dir+"/__init__.py")

    except PermissionError:
        print(Fore.RED + "Permission Error: " + Fore.RESET +" user " + Fore.YELLOW +
              '{}'.format(getuser()) + Fore.RESET + "has insufficient privilages to create directories.")

    finally:
        print(Fore.RESET)
        return full_directories


# Override build_py to be able to execute a command
class my_build_py(build_py):
    def run(self):
        """ Initialization Routine for setup.py

        :return: Nothing
        """

        print(Fore.CYAN + "Trying to install packages: {}".format(needed_packages))
        print(Fore.RESET)
        # Install the packages as defined in the needed_packages list
        install_packages(needed_packages)

        create_subdirectories(needed_directories)
        '''
        directories = {"config_files": "src/configuration_files",
                       "models": "models",
                       "logdir": "models/logs",
                       "cache_dir": "cache",
                       "message_cache": "cache/messages",
                       "label_cache": "cache/labels",
                       "list_cache": "cache/lists"}
        '''
        
        try:
            # We will create keys.py
            with open("{}/keys.py".format(needed_directories['config_files']), 'w') as key_file:
                email = input("Enter your gmail account: ").replace(" ", "")
                key_file.write("# This file was automatically generated by {}\n".format(__file__))
                key_file.write("user_id = \"" + (email if '@' in email else email + "@gmail.com") + "\"\n")

                print("Setting variables defined in needed_directories")
                
                # iterate through the keys and values in the needed_directories dict and set them as variables
                for variable, path in needed_directories.items():

                    # Actually write to the key_file the variable name and the value we give it
                    key_file.write('{} = "{}/{}"\n'.format(variable, os.getcwd(), path))

                    print(Fore.GREEN + "Set " + Fore.CYAN + variable + Fore.GREEN +
                          " to '" + Fore.BLUE + path + Fore.RESET + "'") 

                print(Fore.GREEN + "Created " + Fore.BLUE + "{}/keys.py".format(needed_directories['config_files']) + Fore.RESET)
                
                
                
        except PermissionError as PE:
            print(Fore.RED + "Error: " + Fore.RESET + "user '" + Fore.RED + getuser() + Fore.RESET +
                  "' has insufficient privilages to create files")

        finally:
            print(Fore.RESET)
            build_py.run(self)

        

setup(
    name='HappyMail',
    version='0.1',
    packages=['src',],
    license='MIT License',
    long_description=open('README.md', mode='r').read(),
    cmdclass={'build_py': my_build_py}
)
