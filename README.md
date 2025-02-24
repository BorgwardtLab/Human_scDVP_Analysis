# Human scDVP data analysis

Data analysis scripts for "Spatial single-cell proteomics enables continuous mapping of protein gradients in human liver tissues".

### Installation

To install the software, first create an isolated conda environment, and add [poetry](https://python-poetry.org/) (our dependency manager of choice) to its dependencies:

```bash
conda create -n humanscDVP python=3.11 poetry
conda activate humanscDVP
```

Navigate to the repository folder, and run:

```bash
poetry install
```

In order to convert mouse gene names to their human orthologs, you will also need to install the [pyorthomap](https://github.com/vitkl/orthologsBioMART) package from GitHub:

```bash
pip install git+https://github.com/vitkl/orthologsBioMART.git
```

### Usage

To reproduce the output provided in `results`, run the following sequence of commands:

1. Run LMM analysis for control human samples:

```bash
python LMM_analysis.py --data_path ../data/human/ --patient_group 1 --plot --save --species human
```

2. Run LMM analysis for fibrotic human samples:

```bash
python LMM_analysis.py --data_path ../data/human/ --patient_group 2 --plot --save --species human
```

3. Run LMM analysis for mouse samples:

```bash
python LMM_analysis.py --data_path ../data/mouse/ --patient_group 3 --plot --save --species mouse
```

4. Compare slopes between control and fibrotic human samples:

```bash
python Compare_slopes.py --control_model_results ../results/human_controls/results_1_cutoff\=0.7.tsv --treatment_model_results ../results/human_fibrotic/results_2_cutoff\=0.7.tsv --plot --one_sided
```