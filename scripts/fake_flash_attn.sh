#!/bin/bash

# 1) create a small local stub package
mkdir -p ~/fake_flash_attn/flash_attn
cat > ~/fake_flash_attn/flash_attn/__init__.py <<PY
# tiny stub to satisfy imports; functions will raise if actually called
__all__ = ["flash_attn_func", "flash_attn_varlen_func"]
def flash_attn_func(*args, **kwargs):
    raise RuntimeError("flash_attn stub: flash_attn is not installed")
def flash_attn_varlen_func(*args, **kwargs):
    raise RuntimeError("flash_attn stub: flash_attn is not installed")
PY

cat > ~/fake_flash_attn/setup.py <<PY
from setuptools import setup, find_packages

setup(
    name="flash-attn",
    version="0.0.0",
    packages=find_packages(),
)
PY

# 2) make an editable install so Python can import it
pip install -e ~/fake_flash_attn