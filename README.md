# RoGER: Graph-based Recommendation with Review Enrichment

This is the official implementation of our graph-based recommender system, RoGER, which leverages both user-item interaction graphs and textual reviews to enrich recommendations.

The code is built on top of the [Elliot](https://github.com/sisinflab/elliot) framework. Please refer to the [documentation](https://elliot.readthedocs.io/en/latest/) for details on the framework and experiment management.

## Models

The main model files are:

- [external/models/roger/RoGER.py](external/models/roger/RoGER.py)
- [external/models/roger/RoGERModel.py](external/models/roger/RoGERModel.py)

## Installation

We recommend using a Python virtual environment or Conda environment. All dependencies are listed in `requirements.txt` and `rmg_env.yml`.

```sh
# Using Conda
conda env create -f rmg_env.yml
conda activate rmg
```

## Datasets

Datasets should be placed in the `data/` folder. Each dataset should include train, validation, and test splits, as well as review-based side information.

Example dataset structure:
```
data/
  Office_Products/
    office_Train_final.tsv
    office_Val_final.tsv
    office_Test_final.tsv
    office_interactions.tsv
    1/  # review features folder with an .npy file for each train interaction
```

To access the datasets, visit the following SharePoint link: [Datasets on SharePoint](https://politecnicobari-my.sharepoint.com/:f:/g/personal/claudio_pomo_poliba_it/EheHLjfdlhRIvX4mGr1VnLABE6GAavuyEaje6YV0gF4VJQ?e=ByGVwg)

## Configuration

All experiments are configured via YAML files in the `config_files/` directory. Each file specifies the dataset, model, hyperparameters, and evaluation metrics.

Example config: [`config_files/Office_Products.yml`](config_files/Office_Products.yml)

## Running Experiments

To run an experiment, use:

```sh
python start_experiments.py --config Office_Products
```

This will execute the experiment defined in `config_files/Office_Products.yml`. Results and logs will be saved in the `results/` and `log/` folders.

## Citation

If you use RoGER in your research, please cite our work:

```
@inproceedings{<your-citation>}
  ...
}
```

## Contact

For questions or issues, please open a GitHub issue or contact the authors.

---

**Note:** For details on datasets and review features, see the paper or contact the authors.