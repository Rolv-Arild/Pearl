from setuptools import setup, find_packages

exec(open('rlgym_tools/version.py').read())

with open('README.md', 'r') as readme_file:
    long_description = readme_file.read()

setup(
    name='pearl',
    packages=find_packages(),
    version="0.1.0",
    description='',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Rolv-Arild Braaten',
    url='https://rlgym.github.io',
    install_requires=[
        'rlgym>=2.0.0rc0',
    ],
    python_requires='>=3.8',
    license='Apache 2.0',
    license_file='LICENSE',
    keywords=['rocket-league', 'gym', 'reinforcement-learning'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        "Operating System :: Microsoft :: Windows",
    ],
)