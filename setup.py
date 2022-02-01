from setuptools import setup
from setuptools import find_packages


def _requires_from_file(filename):
    return open(filename).read().splitlines()


with open('README.rst') as f:
    readme = f.read()


setup(
    name="si4ul",
    package_dir={"": "src"},
    packages=find_packages("src"),
    version="0.1.5",
    url="https://github.com/takeuchi-lab/si4ul",
    author="Takeuchi Lab",
    author_email="omori.y.mllabl.nit@gmail.com",
    description="selective inference for unsupervised learning",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    install_requires=_requires_from_file('requirements.txt'),
    long_description=readme,
    license="MIT License",

    
    # py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
    # include_package_data=True,
    # zip_safe=False,
)
