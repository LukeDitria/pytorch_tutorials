<img src="datasets/banner.png" alt="Pytorch Tutorials" width="800"/>

# Beginner Level Deep Learning Tutorials in Pytorch! <br>
Note that these tutorials expect some knowledge of deep learning concepts. While some of the concepts are explained we are mainly focusing on (in detail) how to implement them in python with Pytorch.<br>
I have compiled a list of additional resources that cover many of the concepts we look at, the YouTube series section are incredibly valuable!<br>

[Deep learning google sheets](https://docs.google.com/spreadsheets/d/1WNJmgsVrLqH522yQ47euqAuO83a4WvJe/edit?usp=sharing&ouid=115240163501200760663&rtpof=true&sd=true)<br>
If you have any good resources let me know and I can add them!<br>
If you can't find an explanation on something you want to know let me know and i'll try to find it!<br>
<br>
<b>Some level of basic Python programming knowledge is expected.</b><br>
<b>More sections to come! </b><br>
<b>Let me know if you want to see anything else! </b><br>

## Help support this work!
<b> Donate here! </b> <br>

https://www.buymeacoffee.com/lukeditria
</br>

## Corresponding Videos
[Pytorch Youtube Playlist](https://youtube.com/playlist?list=PLN8j_qfCJpNhhY26TQpXC5VeK-_q3YLPa&si=bMjdMvuVIX8X0yTz)<br>
[Reinforcement Learning Youtube Playlist](https://youtube.com/playlist?list=PLN8j_qfCJpNg5-6LcqGn_LZMyB99GoYba&si=1HVWNHNQOhw2GrYq)<br>

Let me know if you want to see a video on any particular section!

## Discord Server
Get help in my [Discord Server](https://discord.gg/8g92X5hjYF)<br>

## Installing
These notebooks need Python 3.10 or newer. Let's set up a virtual environment so everything stays in one place and doesn't mess with anything else on your computer!<br>

```
git clone https://github.com/LukeDitria/pytorch_tutorials.git
cd pytorch_tutorials
python -m venv .venv
```

Now activate it (you'll need to do this every time you open a new terminal)<br>

```
# Linux / Mac
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

Then install the libraries and start Jupyter!<br>

```
pip install -r requirements.txt
jupyter lab
```

A few things that might trip you up:<br>
* If you have an NVIDIA GPU, make sure you install the version of PyTorch that matches your CUDA version. The [PyTorch install page](https://pytorch.org/get-started/locally/) will give you the right command, run it before `pip install -r requirements.txt`. Newer GPUs (like the RTX 50 series) need a recent CUDA build.<br>
* The notebooks that use `torchtext` (Sections 12, 13 and 14, plus Semantic Clustering) were made with an older version of PyTorch as `torchtext` is no longer being updated. For those, make a second virtual environment with Python 3.11 and run `pip install torch==2.1.0 torchtext==0.16.0 "numpy<2" pandas tqdm portalocker` as well as the other libraries you need (`torchtext` will install `torchdata` for you).<br>
* The Procgen notebook (Section 11) needs `procgen` which doesn't install on newer versions of Python, you'll need an older environment for it too.<br>
* A few notebooks need a dataset you have to download yourself, the notebook will tell you where to get it and where to put it (Section 8 needs CUB-200 from Kaggle, Section 14's image captioning needs COCO).<br>
* Some of the notebooks take a very long time to train, they were run on a big GPU for hours (or days!). You don't need to run them to the end to follow along, but you may want to lower the number of epochs.<br>

## Contents (So Far!)
Section 0 -> Python basics that will be expected knowledge<br>
Section 1 -> Implementing some basic Machine Learning Algorithms in Python with Numpy<br>
Section 2 -> Pytorch intro and basics, basic Machine Learning Algorithms with Pytorch<br>
Section 3 -> Multi-Layer Perceptron (MLP) for Classification and Non-Linear Regression<br>
Section 4 -> Pytorch Convolutions and CNNs <br>
Section 5 -> Pytorch Transfer Learning <br>
Section 6 -> Pytorch Tools and Training Techniques <br>
#### Applications + Advanced
Section 7 -> Pytorch Autoencoders and Representation Learning <br>
Section 8 -> Pytorch Bounding Box Detection and Image Segmentation <br>
Section 9 -> Pytorch Image Generation <br>
Section 10 -> Pytorch Trained Model Interpretation <br>
Section 11 -> Pytorch Reinforcement Learning <br>

#### Sequential Data
Section 12 -> Using Sequential Data <br>
Section 13 -> All about Attention <br>
Section 14 -> Transformer Time <br>

## Contents (In Progress!)
Section 15 -> Deploying Models <br>
Section 16 -> Advanced Applications <br>

## Contents (To Come!)

## Folder layout:
notebooks -> Tutorials and Skeleton code (Start here)<br>
solutions -> Skeleton code Solutions<br>
data -> Data and Images<br>
