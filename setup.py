from setuptools import setup, find_packages

setup(
    name="urbex",
    version="1.0.0",
    description="Urbex - agentic city-scale 3D visual grounding with RL-trained tool use",
    packages=find_packages(include=["roll", "roll.*"]),
    python_requires=">=3.10",
)
