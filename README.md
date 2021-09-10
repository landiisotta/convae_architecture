# ConvAE architecture for latent representations of EHR sequences

This folder includes the ConvAE architecture implementation and an example dataset to 
learn patient representations from their EHRs as described in [1]. 

### Technical Requirements

```
Python 3.6+

```

# Run ConvAE
Download the `convae_architecture` folder

```bash
$ git clone http://github.com/landiisotta/convae_architecture
```

The full list of required Python Packages is available in `requrirements.txt` file. It is possible
to install all the dependencies by:

```bash
$ pip install -r requirements.txt 
```

Examples of randomly generated input EHRs (train and test) can be found in `data_example` folder. 
Outputs will be stored in `./data_example/encodings` and include ConvAE latent representations, 
EHR sequences organized in subsequences of desired length, and best model weights.

To train the model:

```bash
sh learn_patient_representations.sh
``` 

To test the representations learned on the test set:

```bash
sh learn-patient-representations.sh test
```

# Data 
Synthetic data include:
> 200 patients, 50:50 split for train and test;

> vocabulary size = 200;

> min sequence length = 3;

> max sequence length = 100;

> embedding dimension = 100;

> subsequence length = 32.

Model parameters can be modified in `utils.py`. 
This example randomly initialize the embedding matrix.

[1] Landi, I., Glicksberg, B. S., Lee, H. C., Cherng, S., Landi, G., Danieletto, M., Dudley, J. T., Furlanello, C., & Miotto, R. Deep representation learning of electronic health records to unlock patient stratification at scale. npj Digit. Med. 3, 96 (2020). https://doi.org/10.1038/s41746-020-0301-zDeep
