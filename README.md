# Bivariate Causal Discovery using Bayesian Model Selection

[Paper on Arxiv](https://arxiv.org/abs/2306.02931)

Installation instructions:
Simply run `pip install -r requirements.txt`
This will run the setup script and install all the required dependencies.

Structure:
- bin/ contains scripts to run the main experiments with real and synthetic data (Table 1)
- experiments/ contains additional experiments
- gplvm_causal_discovery/ contains the main code
    - data/ contains the data files as well as aditional modules
    - models/ contains the GPLVM models
        - BayesGPLVM: Unsupervised closed form GPLVM
        - PartObsBayesianGPLVM: Conditional closed form GPLVM
        - GeneralisedUnsupGPLVM: Unsupervised stochastic GPLVM (large data)
        - GeneralisedGPLVM: Conditional stochastic GPLVM (large data)
    - results/ contains all the results of the experiments
    - train_methods/ contains the scripts required for training the methods
    - utils.py contains the utilities

To print the results in the paper, run the notebook in notebooks/paper_results.ipynb

To run the method on data, run the script bin/synthetic_real.py

### Latent Gaussian Process Models:
Performance was found to increase with better inference, better initialisation of hyperparameters, as well as inclusion of more kernels. As this is an active area of research, we expect implementation of any improvements in these to help.

### Random Restarts:
The overarching principle behind the method is to choose the model with highest marginal likelihood (or lower bound to it). As latent GP models are susceptible to local optima, random restarts with different initialisation is recommended.

