import os

from setuptools import setup, find_packages


with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))

    with open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()

def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            return line.split("'")[1]

    raise RuntimeError('Unable to find version string.')

with open('requirements.txt', 'r') as requirements:
    setup(
        name='scnet',
        version=get_version('scnet/__init__.py'),
        install_requires=list(requirements.read().splitlines()),
        packages=find_packages(),
        description=
            'library implementing self-correcting neural networks',
        python_requires='>=3.6',
        author='Klas Leino',
        author_email='kleino@cs.cmu.edu',
        classifiers=[
            'Programming Language :: Python :: 3',
            'License :: OSI Approved :: MIT License',
            'Operating System :: OS Independent'],
        long_description=long_description,
        long_description_content_type='text/markdown'
    )
