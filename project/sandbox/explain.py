import shap
import torch
from predict import model, tokenizer

text = ["I feel hopeless and tired all the time"]

inputs = tokenizer(text, return_tensors="pt")
explainer = shap.Explainer(model)
shap_values = explainer(inputs["input_ids"])

shap.plots.text(shap_values)
