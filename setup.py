from setuptools import setup, find_packages

classifiers = [
    'Development Status :: 1 - Planning',
    'Intended Audience :: Education',
    'Operating System :: OS Independent',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.9'
]

setup(
    name='nest-synelprof',
    version='0.1.0',
    author='betalab group UNIPD',
    author_email='enrico.prataviera@unipd.it',
    packages=find_packages(),
    scripts=[],
    url='https://github.com/BETALAB-team/NEST-SynElProf',
    license='LICENSE',
    description='This repository is a small python library to create, for an Italian case study, synthetic electric consumption profiles for residential dwelling.',
    install_requires=[
        "matplotlib", 
        "scipy", 
        "numpy",
        "pandas",
        "pyarrow",
        "fastparquet",
    ],
)
