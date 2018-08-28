from subprocess import call
from sys import executable
from distutils.core import setup
from distutils.command.build_py import build_py
import os
from os.path import exists


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
        print("installing package {} with pip" .format(package))
        pip_command = "{} -m pip install {} --user".format(executable, package)
        print("Running {}".format(pip_command))
        retcode = call(pip_command.split(' '))

        if retcode is not 0:
            print("return code was {} when trying to install {}".format(retcode, packages))


# Override build_py to be able to execute a command
class my_build_py(build_py):
    def run(self):
        print("Trying to install packages: {}".format(needed_packages))

        install_packages(needed_packages)

        # Now we actually create the config files
        print("Creating config files")

        # We set the directory to the configuration
        configdir = "{}/src/configuration_files".format(os.getcwd())
        # If the configuration file directory doesn't exist
        if not exists(configdir):
            os.mkdir(configdir)
            with open("{}/__init__.py".format(configdir), 'w') as outfile:
                outfile.write("# Generated by {}\n".format(__file__))

        # If the keys.py file doesn't exist
        if not exists("{}/keys.py".format(configdir)):
            # We can create it
            with open("{}/keys.py".format(configdir), 'w') as key_file:
                email = input("Enter your gmail account: ").replace(" ", "")
                key_file.write("# This file was automatically generated by {}\n".format(__file__))
                key_file.write("user_id = \"" + (email if '@' in email else email + "@gmail.com") + "\"\n")

        build_py.run(self)


setup(
    name='HappyMail',
    version='0.1',
    packages=['src',],
    license='MIT License',
    long_description=open('README.md', mode='r').read(),
    cmdclass={'build_py': my_build_py}
)
