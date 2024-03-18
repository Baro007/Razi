Râzî
<p align="center">


DISCLAIMER - Do not take any advice from Râzî seriously yet. This is a work in progress and taking any advice seriously could result in serious injury or even death. 


</p>

## Overview
Râzî is a Large Language Model that has the potential to pass the US Medical Licensing Exam. This open-source endeavor aims to give everyone access to their own private assistant with medical knowledge. Modeled after Meta's [Llama2](https://ai.meta.com/llama/) 7 billion parameter Large Language Model, Râzî has been refined through tuning on a substantial Medical Dialogue Dataset and further enhanced with the use of Reinforcement Learning & Constitutional AI techniques. At just 3 Gigabytes, Râzî is designed for local device deployment, obviating the need for a paid API. The model is built for offline use to ensure patient confidentiality is maintained and is compatible with iOS, Android, and Web platforms. Contributions to its development through pull requests are welcomed and encouraged.

## Dependencies
- [Numpy](https://numpy.org/install/) (Use matrix math operations)
- [PyTorch](https://pytorch.org/) (Build Deep Learning models)
- [Datasets](https://huggingface.co/docs/datasets/index) (Access datasets from huggingface hub)
- [Huggingface_hub](https://huggingface.co/docs/huggingface_hub/v0.5.1/en/package_reference/hf_api) (Access huggingface data & models)
- [Transformers](https://huggingface.co/docs/transformers/index) (Access models from HuggingFace hub)
- [Trl](https://huggingface.co/docs/trl/index) (Transformer Reinforcement Learning and fine-tuning)
- [Bitsandbytes](https://github.com/TimDettmers/bitsandbytes) (Model size optimization through 'quantization')
- [Sentencepiece](https://github.com/google/sentencepiece) (Byte Pair Encoding scheme '\tokenization')
- [OpenAI](https://openai.com) (Creation of synthetic fine-tuning and reward model data)
- [TVM](https://tvm.apache.org/) (Tensor Virtual Machine for onnx model conversion for efficient cross-platform use)
- [Peft](https://huggingface.co/blog/peft) (Parameter Efficient Fine Tuning through low rank adaption (LoRa))
- [Onnx](https://onnx.ai/) (Conversion of the trained model to a universal format)

## Installation
You can install all dependencies at once with [pip](https://pip.pypa.io/en/stable/installation/):

```bash
pip install numpy torch datasets huggingface_hub transformers trl bitsandbytes sentencepiece openai tvm peft onnx
```

## iOS Quickstart v2

1. Clone the repository
```bash
git clone https://github.com/your-username/Razi
```
2. Download the Weights
```bash
mkdir -p dist/prebuilt
git clone https://github.com/mlc-ai/binary-mlc-llm-libs.git dist/prebuilt/lib
cd dist/prebuilt
git lfs install
wget --no-check-certificate 'https://drive.google.com/file/d/1-your-google-drive-id/view?pli=1'
cd ../..
```
3. Build the Tensor Virtual Machine Runtime
```bash
git submodule update --init --recursive
pip install apache-tvm
cd ./ios
pip install --pre --force-reinstall mlc-ai-nightly mlc-chat-nightly -f https://mlc.ai/wheels
./prepare_libs.sh
```
** Find the right version of MLC LLM for your system [here](https://mlc.ai/package/)

4. Add Weights to Xcode
```bash
cd ./ios
open ./prepare_params.sh # ensure the builtin_list contains only the models relevant to Râzî
./prepare_params.sh
```
5. Open the Xcode Project and run

## DIY Training
To train Râzî, utilize the provided training notebook locally or through a cloud service like Google Colab Pro:

#### Cloud Training
Use the following link to access the notebook:
https://colab.research.google.com/github/your-username/Razi/blob/main/llama2.ipynb

#### Local Training
Clone the repository and start the Jupyter notebook:
```bash
git clone https://github.com/your-username/Razi.git
jupyter training.ipynb
```
Find Jupyter [here](https://jupyter.org/install).

## Usage
For information on utilizing Râzî, visit the corresponding repository: https://huggingface.co/your-username/razi

## Credits
This project is developed with contributions from multiple parties including Meta, MedAlpaca, Apache, MLC Chat & OctoML.
