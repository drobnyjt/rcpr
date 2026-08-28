from setuptools import setup
from setuptools_rust import Binding, RustExtension

setup(
    name="rcpr",
    version="0.5.0",
    rust_extensions=[
        RustExtension(
            "pyacpr",
            binding=Binding.PyO3,
            features=["python"]
        )
    ],
    # rust extensions are not zip safe, just like C-extensions.
    zip_safe=False,
)