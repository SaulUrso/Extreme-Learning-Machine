# Extreme Learning machine

## Code of the algorithms

- `nesterov.py`: Implementation of the Nesterov Accelerated Gradient algorithm.
- `modelutils.py`: Contains the code for the Extreme Learning Machine (ELM) model.
- `cholesky.py` and `backfwd.py`: Provide the code for solving the closed-form solution using Cholesky decomposition and forward/backward substitution.


## Setting Up the Python Environment

Make sure Python 3.11 is installed. If not, download it from the [official website](https://www.python.org/downloads/) or use a package manager (e.g., `brew install python@3.11` on macOS).

To create and use a virtual environment:

1. **Create a new virtual environment:**
    ```bash
    python3.11 -m venv .venv
    ```

2. **Activate the environment:**
    - On macOS/Linux:
      ```bash
      source .venv/bin/activate
      ```
    - On Windows:
      ```cmd
      .venv\Scripts\activate
      ```

3. **Install required packages** (replace `requirements.txt` with your dependencies file if available):
    ```bash
    pip install -r requirements.txt
    ```

## Experiments 

With the environment activated, you can use Jupyter to run the notebooks `nesterov.ipynb` and `cholesky.ipynb`:

```bash
pip install jupyter
jupyter notebook
```

Then, open the notebooks in your browser and execute the cells as needed.