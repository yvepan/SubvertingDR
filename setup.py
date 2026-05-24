from setuptools import find_packages, setup

LATEST_VERSION = "0.1.0"

exclude_packages = [
    "selenium",
    "webdriver",
    "fastapi",
    "fastapi.*",
    "uvicorn",
    "jinja2",
    "langgraph"
]

with open(r"README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r") as f:
    reqs = [line.strip() for line in f if not any(pkg in line for pkg in exclude_packages)]

setup(
    name="forge-deep-research",
    version=LATEST_VERSION,
    description="Self-contained FORGE deep-research experiment artifact with vendored GPT Researcher testbed",
    package_dir={'gpt_researcher': 'gpt_researcher'},
    packages=find_packages(exclude=exclude_packages),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://anonymous.4open.science/r/FORGE-6A67",
    author="FORGE Deep Research Artifact Authors",
    license="Responsible Research Use License",
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires='>=3.11',
    install_requires=reqs,


)
