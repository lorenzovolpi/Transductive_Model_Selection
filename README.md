# Transductive Model Selection

## Setup

To setup the environment for running the experiments, first clone the repository:

```bash
git clone https://github.com/lorenzovolpi/tms.git tms --depth 1
cd tms
```
then create the python virtual environment and install requirements:

```bash
python -m venv .venv
chmod +x .venv/bin/activate
source ./.venv/bin/activate
python -m pip install -e .
```

## Run

### Experiments

To run the experiments shown in the cited paper, run:

```bash
python -m transd
```

To control the desired setting to run the experiments (i.e., whether `binary` or `multiclass`) change the value of the variable `PROBLEM` in the file `leap/exp/env.py`.

### Plots and Tables

To generate the plots run:

```bash
python -m plot
```

To generate the tables run:

```bash
python -m table
```

### Output

The `output/tms` folder will contain the subfolders:
- `transd` with the results for the main experiments;
- `plots/transd` containing all the generated plots;
- `tables` containing both the `.pdf` and the `.tex` files of the generated tables.

To change the output root folder, create an `env.json` file with the following structure:
```json
{
    "global": {
        "OUT_DIR": "my/custom/outdir"
    }
}
```
before running the above commands.

### Parallel execution

The execution of the tests is parallelised, with each combination of (`classifier`, `dataset`, `method`)
running in a separate process. The `N_JOBS` field can be set in `env.json` to control the number
of availables processors to dedicate to the execution:
```json
{
    "global": {
        "N_JOBS": 16
    }
}
```

