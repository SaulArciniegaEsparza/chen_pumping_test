from setuptools import setup

setup(
    name='chen_pumping_test',
    version='1.0.0',
    packages=['chen', 'chen.models', 'chen.models.pumping_test', 'chen.models.interpretation', 'chen.structure',
              'chen.utilities'],
    url='',
    license='BSD',
    author='Saul Arciniega Esparza',
    author_email='zaul.ae@gmail.com',
    description='Pumping test analysis for parameter estimation and drawdown interpretation',
    classifiers=['Topic :: Civil Engineer ::  Hydrology :: Physics']
)
