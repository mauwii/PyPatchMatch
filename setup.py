from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as readme_file:
    readme = readme_file.read()

requirements = [
    "numpy",
    "pillow",
    "tqdm"
]

setup(
    name='pypatchmatch',
    packages=['patchmatch'],
    version='0.1.4',
    url='https://github.com/invoke-ai/PyPatchMatch',
    python_requires='>=3.9',
    install_requires=requirements,
    description='This library implements the PatchMatch based inpainting algorithm.',
    long_description=readme,
    long_description_content_type="text/markdown",
    package_data={'patchmatch': ['Makefile','csrc/*','build/*']},
)
