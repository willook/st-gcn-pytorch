from setuptools import setup, find_packages

setup(
    name='stgcn',
    version='1.0.0',
    author='willook',
    author_email='jongwhoa.lee@gmail.com',
    url='https://github.com/willook/st-gcn-pytorch',
    classifiers=[
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3 :: Only'
    ],
    packages=find_packages(),
    install_requires=[
        'torch',
    ],
    python_requires='>=3.8',
)