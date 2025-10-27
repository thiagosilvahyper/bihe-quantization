from setuptools import setup, find_packages

setup(
    name="bihe-quantization",
    version="1.0.0",
    description="BIHE: Next-generation vector quantization",
    author="Thiago Ferreira da Silva",
    author_email="pt.thiagosilva@gmail.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.11",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "torch>=2.0.0",
        "sentence-transformers>=2.2.0",
    ],
)
