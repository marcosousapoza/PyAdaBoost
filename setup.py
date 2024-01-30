from setuptools import setup, find_packages

setup(
    name='adapy',
    version='0.1.0',
    author='Marco Sousa-Poza',
    author_email='marco.sousapoza@gmail.com',
    packages=find_packages(),
    description='An implementation of the AdaBoost algorithm',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=[
        'numpy',
        'scikit-learn',
        'matplotlib'
    ],
)