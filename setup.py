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

    :param list directories: List of subdirectories within the current directory to create
    :param str base_dir: Path to the base directory under which all the directories will be installed.\
     this parameter *should* be an absolute path for universal use.
    :return: A list of the directories but with the base_dir prepended to them
    """

    # Strip the tailing / symbol from the base directory for consistency
    base_dir = base_dir.rstrip('/')

    full_directories = []

    # Try to create the subdirectories
    try:

        # For each directory in the given directories list
        for directory in directories:

            # Saves the full path to the directory and strips it on the right for consitency
            fixed_dir = base_dir + '/' + directory.rstrip('/')

            # Save it to the full_directories list
            full_directories.append(fixed_dir)

            # If the directory doesn't already exist, then we create it
            if not exists(full_dir):

                print(Fore.YELLOW +  full_dir + " doesn't exist")

                # Call the makedirs function to create all the directories in between
                os.makedirs(full_dir)

                print(Fore.GREEN + "Created directory " + full_dir + Fore.RESET)

            if not exists(full_dir + "/__init__.py"):
                # Create the __init__.py file within the directories
                with open(full_dir + "/__init__.py", "w") as init:
                    init.write("# This file was generated by {}\n".format(__file__))

                print(Fore.GREEN + "Created "+full_dir+"/__init__.py")
            print(Fore.RESET)

    except PermissionError:
        print(Fore.RED + "Permission Error: " + Fore.RESET +" user " + Fore.YELLOW +
              '{}'.format(getuser()) + Fore.RESET + "has insufficient privilages to create directories.")

    return full_directories


# Override build_py to be able to execute a command
class my_build_py(build_py):
    def run(self):
        """ Initialization Routine for setup.py

        :return:
        """

        print(Fore.CYAN + "Trying to install packages: {}".format(needed_packages))
        print(Fore.RESET)
        # Install the packages as defined in the needed_packages list
        install_packages(needed_packages)

        try:

            # Create directory for storing data caches
            cache_dir = "{}/cache".format(os.getcwd())
            if not exists(cache_dir):
                print(Fore.YELLOW + "Cache directory not found")
                os.mkdir(cache_dir)
                print(Fore.GREEN + "Created {}".format(cache_dir))

            else:
                print(Fore.GREEN + "Found cache directory: {}".format(cache_dir))

            print(Fore.RESET)

            # Create __init__ file for src/
            source_dir = "{}/src".format(os.getcwd())

            if not exists("{}/__init__.py".format(source_dir)):
                with open("{}/__init__.py".format(source_dir), 'w') as initfile:
                    initfile.write("# Generated by {}\n".format(__file__))

                print(Fore.GREEN + "Created {}/__init__.py".format(source_dir))
            # Now we actually create the config files
            print(Fore.RESET+"Creating config files")

            # We set the directory to the configuration
            configdir = "{}/src/configuration_files".format(os.getcwd())
            # If the configuration file directory doesn't exist
            if not exists(configdir):
                print(Fore.YELLOW + "Couldn't find {}, making".format(configdir))
                os.mkdir(configdir)
                print(Fore.GREEN + "Created {}".format(configdir))

            else:
                print(Fore.GREEN + "Found directory {}".format(configdir))

            print(Fore.RESET)

            if not exists("{}/__init__.py".format(configdir)):

                with open("{}/__init__.py".format(configdir), 'w') as outfile:
                    outfile.write("# Generated by {}\n".format(__file__))

                print(Fore.GREEN + "Created {}/__init__.py".format(configdir))

            else:
                print(Fore.GREEN + "Found {}/__init__.py".format(configdir))

            print(Fore.RESET)

            # If the keys.py file doesn't exist
            if not exists("{}/keys.py".format(configdir)):
                print(Fore.YELLOW + "{}/keys.py doesn't exist".format(configdir))
                print(Fore.RESET)

                # We can create it
                with open("{}/keys.py".format(configdir), 'w') as key_file:
                    email = input("Enter your gmail account: ").replace(" ", "")
                    key_file.write("# This file was automatically generated by {}\n".format(__file__))
                    key_file.write("user_id = \"" + (email if '@' in email else email + "@gmail.com") + "\"\n")
                    key_file.write('cache_dir = "{}"\n'.format(cache_dir))

                print(Fore.GREEN + "Created {}/keys.py")
            else:
                print(Fore.GREEN + "Found {}/keys.py".format(configdir))

        except PermissionError as PE:
            print(Fore.RED + PE)

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
