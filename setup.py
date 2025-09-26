from setuptools import setup
from setuptools_rust import Binding, RustExtension

setup(
    name="rcpr_py",
    version="0.0.1",
    rust_extensions=[
        RustExtension(
            "rcpr_py",
            binding=Binding.PyO3,
        )
    ],
    # rust extensions are not zip safe, just like C-extensions.
    zip_safe=False,
)