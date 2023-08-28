# Fuzzy Metaballs+: Differentiable Renderering with 3D Gaussians, Flow and More. 
# [Project Page](https://leonidk.github.io/fmb-plus/)
This is an expanded version of the original [FM renderer](https://leonidk.github.io/fuzzy-metaballs/) with support for flow, mesh exporting, 2-parameter and zero-parameter renderers. 

It primarily is useful for reconstructing CO3D Sequences. The generation operation is

* Git clone [unimatch](https://github.com/autonomousvision/unimatch) for generating optical flow and  [XMem](https://github.com/hkchengrex/XMem) for propogating the first mask into the same root directory.
* Fetch some CO3D sequences. For example using `get_co3d.sh` for the single sequence teddy bear. 
* Run `generate_inputs.ipynb` to generate flows and masks for the reconstruction
* Run `run_co3d_sp.ipynb` or `run_co3d_sp-zpfm.ipynb` to run the reconstructions with either the two parameter or zero parameter models
* Compile [PoissonRecon](https://github.com/mkazhdan/PoissonRecon) and run it to generate a mesh via `PoissonRecon --in tmp_out/teddybear_34_1479_4753.ply --out teddy.ply --bType 2 --depth 6`

## TODO

* Clean up code so it's easier to run non-CO3D sequences
* Add an importer from the released version of 3DGS format