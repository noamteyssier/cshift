from setuptools import setup

setup(
    name="cshift",
    version="0.1.0",
    description="A tool to perform cluster enrichment/depletion analyses",
    author="Noam Teyssier",
    author_email="Noam.Teyssier@ucsf.edu",
    packages=["cshift"],
    install_requires=["numpy", "scipy", "pandas", "seaborn", "matplotlib"],
)
