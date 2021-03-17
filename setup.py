from setuptools import setup

setup(
    name = "noiseceiling",
    version = "0.1",
    author = "Lukas Snoek",
    author_email = "lukassnoek@gmail.com",
    description = "Computing noise ceilings for ML models",
    license = "BSD3",
    keywords = "ML noiseceiling",
    packages=['noiseceiling'],
    install_requires=[
        'numpy',
        'sklearn',
        'pandas'
    ]
)