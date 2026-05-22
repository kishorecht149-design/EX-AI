from setuptools import setup, find_packages

setup(
    name="streamlit",
    version="99.9.9",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "streamlit=mock_streamlit.main:main",
        ],
    },
)
