from setuptools import setup, find_packages

setup(
    name="eartag_jetson",
    version="0.1.0",
    description="Cow ear‑tag detection & recognition on NVIDIA Jetson",
    author="Trieu Tran",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    package_data={
        "eartag_jetson.resources": ["*"],
    },
    entry_points={
        "console_scripts": [
            "eartag = eartag_jetson.cli:main",
        ],
    },
)
