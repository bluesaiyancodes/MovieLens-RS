
## Full Implementation of GHRS: Graph-based Hybrid Recommendation System

Paper  [Link](https://doi.org/10.1016/j.eswa.2022.116850) \
Partial Implementation: (official) [Link](https://github.com/hadoov/GHRS)  


## Original Framework

![ghrs structure](/Plots/ghrs-structure.png)

The orginal framework of the recommendation system proposed in the paper.

``Step 1``, ``Step 2`` and ``Step 3`` is provided in the official partial implementation of the paper. **We build the rest** based on the details outlined in the paper.

We recommend going through the slides [here](/slides/Project_final.pptx) first and then get started.

### Directory Structure
![Directory structure](/Plots/dir_structure.png)
``Datasets/1m/ml-1m/`` contains *movies.dat*, *ratings.dat* and *users.dat* files

``Datasets/1m/ml-100k/`` contains *u.data*, *u.user, ...*  files

### Impplementation
1. Setup new virtual environment\
``python -m venv <envname>``\
``source <envname>/bin/activate``

2. Install the required libraries\
``pip install -r requirements.txt``

3. Run PyTorch or Tensorflow implementation\
``python main_pt.py --config config.yaml``\
or\
``python main_tf.py --config config.yaml``


[config.yaml](config.yaml) file contains the experiment parameters and can be modified based on the needs.

## Results

![Experiment Results](/Plots/results.png)


~~~
Implementation performed as a part of the course - Recommendation system @ Kumoh National Institute of Technology, Korea during summer semester of 2024.
~~~

