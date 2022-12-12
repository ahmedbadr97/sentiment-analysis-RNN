import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
setuptools.setup(
    name='sentiment-analysis',
    version='0.0.1',
    author='Ahmed badr',
    author_email='ahmed.k.badr.97@gmail.com',
    description='sentiment analysis model for reviews',
    long_description=long_description,
    long_description_content_type="text/markdown",
    include_package_data=True,
    url='https://github.com/ahmedbadr97/sentiment-analysis-RNN',
    license='MIT',
    packages=['sentiment_analysis'],
    package_dir={
        'sentiment_analysis': 'src/sentiment_analysis'},
    install_requires=['torch', 'gdown'],
)
