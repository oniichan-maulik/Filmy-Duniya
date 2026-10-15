from setuptools import setup, find_packages

setup(
    name="movie_recommender",
    version="0.0.0",
    packages=find_packages(where='.'),
    include_package_data=True,
)
