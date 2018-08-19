from distutils.core import setup
from distutils.command.build_py import build_py
import os
from os.path import exists

# Override build_py to be able to execute a command
class my_build_py(build_py):
    def run(self):
        if not exists("{}/src/configuration_files".format(os.getcwd())):
            os.mkdir("{}/src/configuration_files".format(os.getcwd()))




setup(
    name='HappyMail',
    version='0.1',
    packages=['src',],
    license='MIT License',
    long_description=open('README.md', mode='r').read(),
    cmdclass={'build_py': my_build_py}
)
