from setuptools import setup, find_packages

setup(
    name="weapon-detection-pipeline",
    version="1.0.0",
    description="Modular Real-Time Weapon Detection Framework",
    author="Landry Tiemani",
    author_email="ltiemani@my.harrisburgu.edu",
    url="https://github.com/landrytiemani/weapon-detection-pipeline",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "ultralytics>=8.0.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "PyYAML>=6.0",
        "lap>=0.4.0",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
    ],
)
