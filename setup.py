from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as readme_file:
    readme = readme_file.read()

requirements = [
    "numpy",
    "pillow"
]

setup(
    name='pypatchmatch',
    py_modules=['patchmatch'],
    packages=find_packages(),
    version='0.0.1',
    url='https://github.com/invoke-ai/PyPatchMatch',
    python_requires='>=3.10',
    install_requires=requirements,
    description='This library implements the PatchMatch based inpainting algorithm.',
    long_description=readme,
    long_description_content_type="text/markdown",
)
