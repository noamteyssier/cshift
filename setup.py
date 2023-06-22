from setuptools import setup

exec(open("cshift/__version__.py").read())
setup(
    name="cshift",
    version=__version__,
    description="A tool to perform cluster enrichment/depletion analyses",
    author="Noam Teyssier",
    author_email="Noam.Teyssier@ucsf.edu",
    packages=["cshift"],
    install_requires=["numpy", "scipy", "pandas", "seaborn", "matplotlib", "tqdm"],
)
