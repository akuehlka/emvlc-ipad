from setuptools import setup, find_packages

setup(
    name='antispoofing.mcnns',
    version=open('version.txt').read().rstrip(),
    url='',
    license='',
    author='',
    author_email='allansp84@gmail.com',
    description='',
    long_description=open('README.md').read(),

    packages=find_packages(where='antispoofing.mcnns', exclude=['tests']),

    install_requires=open('requirements.txt').read().splitlines(),

    entry_points={
        'console_scripts': [
            'mcnnsantispoofing.py = antispoofing.mcnns.scripts.mcnnsantispoofing:main',
        ],
    },

)
