import json
import os

def create_notebook(py_filepath, ipynb_filepath, title):
    with open(py_filepath, 'r') as f:
        code = f.read()

    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    f"# {title}\n",
                    "This notebook demonstrates the capabilities of the HAM package."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [line + "\n" for line in code.split("\n")]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.9.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    with open(ipynb_filepath, 'w') as f:
        json.dump(notebook, f, indent=2)
    print(f"Created {ipynb_filepath}")

os.makedirs("examples/notebooks", exist_ok=True)
create_notebook("examples/demo_zermelo.py", "examples/notebooks/demo_zermelo.ipynb", "Zermelo Navigation Demo")
create_notebook("examples/demo_vortex.py", "examples/notebooks/demo_vortex.ipynb", "Vortex Field Navigation Demo")
create_notebook("examples/demo_learned_wind.py", "examples/notebooks/demo_learned_wind.ipynb", "Learned Wind Demo")
create_notebook("examples/demo_discrete_zermelo.py", "examples/notebooks/demo_discrete_zermelo.ipynb", "Discrete Zermelo Demo")
