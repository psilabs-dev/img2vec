from setuptools import setup, find_packages
import os

current_folder = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(current_folder, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='img2vec-pytorch-2',
    version='1.3.0',
    description='Use pre-trained models in PyTorch to extract vector embeddings for any image',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/psilabs-dev/img2vec',
    author='Christian Safka, psilabs-dev',
    author_email='christiansafka@gmail.com, 113860476+psilabs-dev@users.noreply.github.com',
    license='MIT',
    install_requires=[
        'torch',
        'torchvision',
        'numpy'],
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6'
    ],
    keywords='img2vec image vector classification pytorch convert',
    packages=find_packages(),
    python_requires='>=3.6'
)