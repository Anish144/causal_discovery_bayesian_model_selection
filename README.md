# Finding causal direction using Bayesian model selection

Installation instructions:
Simply run `pip -r requirements.txt`
This will run the setup script and install all the required dependencies.

File structure:
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

To get the results in the paper, run the notebook in notebooks/paper_results.ipynb

To run the method on data, run the script bin/synthetic_real.py
