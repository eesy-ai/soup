import setuptools

setuptools.setup(
    # Needed to actually package something
    packages=setuptools.find_packages(exclude=("docs", "examples")),
    # Needed for dependencies
    install_requires=[
        "matplotlib == 3.7",
        "pandas == 2.0",
        "progress == 1.6",
        "pyhrv == 0.4",
        "reportlab == 4.2.2",
    ],
)
