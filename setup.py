from setuptools import setup, find_packages
from typing import List

requirement_file_name = 'requirements.txt'
REMOVE_PACKAGE = '-e .'

def get_requirements()->List[str]:
    with open(requirement_file_name) as requirement_file:
        requirement_list = requirement_file.readlines()
        requirement_list = [ requirement_name.replace("\n", "") for requirement_name in requirement_list]
    
        if REMOVE_PACKAGE in requirement_list:
            requirement_list.remove(REMOVE_PACKAGE)
        return requirement_list
    
    
setup(name='Insurance',
      version='0.0.1',
      description='Insurance prediction',
      author='Sharath Prakash',
      author_email='sharathprakash81@gmail.com',
      packages= find_packages(),
      install_requires=get_requirements()
)