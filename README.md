# GlobalMapper: Arbitrary-Shaped Urban Layout Generation
Official Pytorch Implementation of "GlobalMapper: Arbitrary-Shaped Urban Layout Generation"

[arXiv](https://arxiv.org/abs/2307.09693) | [BibTeX](#bibtex) | [Project Page](https://arking1995.github.io/GlobalMapper/)

This repository contains an educational version of the code for
[GlobalMapper: Arbitrary-Shaped Urban Layout Generation](https://arxiv.org/pdf/2307.09693.pdf).
Every module has been richly annotated to serve as a tutorial for newcomers to
graph-based generative models.

## Environment

The project uses PyTorch and PyTorch Geometric.  A minimal environment can be
created with

```
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-1.9.0+cu111.html
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.9.0+cu111.html
pip install torch-geometric
```

Additional packages such as `matplotlib`, `yaml` and `pickle` are required for
visualisation and configuration.

## Dataset

A small sample dataset (10k blocks) is included as `dataset.tar.gz`.  Unzip it
to `dataset/`.

A larger 120k dataset is available [here](https://purdue0-my.sharepoint.com/:u:/g/personal/he425_purdue_edu/ET2gehuc9BhBhJd_4kIrhbYB0xJNuMDZE6mqVTZd9yDQ3Q?e=AwWMKy).

Each graph in `processed/` contains node attributes describing building
existence, size and position.  The `raw_geo/` directory holds the original
polygons obtained via [osmnx](https://osmnx.readthedocs.io/en/stable/user-reference.html).

## Training

1. Adjust hyper‑parameters in `train_gnn.yaml`.
2. Run

   ```
   python train.py
   ```

   Checkpoints and logs are written to `epoch/` and `tensorboard/` when
   `save_record` is enabled.

## Testing / Reconstruction

Set `dataset_path` and `epoch_name` in `test.py` and run

```
python test.py
```

The script reconstructs blocks from the validation set or samples from the
latent space depending on configuration flags.

## Adding New Attributes

Functional‑zone labels (7 classes) or additional building features can be
incorporated by:

1. **Dataset:**
   - Extend `graph2vector_processed` and `graph_transform` in `urban_dataset.py`
     to extract and append the new attributes to `node_feature`.
   - Update the dimensions of tensors accordingly.
2. **Model:**
   - Increase the input size of `ex_init`/`ft_init` in `model.py` so the network
     consumes the extended feature vectors.
   - Add loss terms in `graph_trainer.py` and weights in `train_gnn.yaml` if the
     new attributes require supervision.

## Canonical Spatial Transformation

`example_canonical_transform.py` demonstrates how raw building polygons are
normalised into the canonical frame used by the dataset.  Consult the
[supplemental material](https://openaccess.thecvf.com/content/ICCV2023/supplemental/He_GlobalMapper_Arbitrary-Shaped_Urban_ICCV_2023_supplemental.pdf)
for mathematical details.

## Visualisation

Simple matplotlib helpers for plotting graphs are provided in `graph_util.py`.

## Citation

If you use this code, please cite:

```text
@InProceedings{He_2023_ICCV,
    author    = {He, Liu and Aliaga, Daniel},
    title     = {GlobalMapper: Arbitrary-Shaped Urban Layout Generation},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {454-464}
}
```

