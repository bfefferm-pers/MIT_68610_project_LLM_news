# NLP-Project
Group project for 6.8610: Quantitative Methods for NLP for Group 5. 

## Setup
```bash
pip install -r requirements.txt
git submodule init 
git submodule update
cd vendors/TransformerLens
pip install .
cd ../hfppl
pip install .
cd ../activations_additions
pip install .
cd ../..
```

### Political Bias Classifier
Download the model and tokenizer from this [link](https://drive.google.com/drive/u/0/folders/1ryvqriRPpTtEoJShn0rz3zhEGW2FJaI_) and update the variables `bias_model_path` and `bias_tokenizer_path` to the absolute path of the model and tokenizer.

Download fine-tuning dataset from this [link](https://drive.google.com/file/d/1PsBFC82nOLeW7qZwWDZE86Zrb0YGTaa5/view?usp=drive_link) and put it in the `political_bias_classifier` folder.
