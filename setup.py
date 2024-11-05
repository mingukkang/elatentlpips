from setuptools import setup, find_packages

setup(
    name='elatentlpips',
    version='0.2',
    description='LatentLPIPS Similarity metric"',
    author='mingukkang',
    author_email='mgkang@postech.ac.kr',
    url='https://github.com/mingukkang/elatentlpips',
    install_requires=['tqdm', "torch>=0.4.0", "torchvision>=0.2.1", "numpy>=1.14.3", "scipy>=1.0.1", "matplotlib>=1.5.1", "diffusers"],
    packages=find_packages(),
    include_package_data=True,
    keywords=['elatentlpips', 'latentlpips', 'perceptual_metric'],
    python_requires='>=3.6',
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
