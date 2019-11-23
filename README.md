## Demo for Visual Question Answering with BUTD

This an user friendly demo for visual question answering. It is essentially a pipeline that combines
[an image feature extractaion tool](https://github.com/peteanderson80/bottom-up-attention) and
[a fast attention implementation](https://github.com/hengyuan-hu/bottom-up-attention-vqa/). 
These two repos implement the BUTD system described in "Bottom-Up and
Top-Down Attention for Image Captioning and Visual Question Answering"
(https://arxiv.org/abs/1707.07998) and "Tips and Tricks for Visual
Question Answering: Learnings from the 2017 Challenge"
(https://arxiv.org/abs/1708.02711).

We modify the BUTD code in both above repos to make it applicable to any new image on the website.
In addition, we re-draw the image to show the attention on the image.
We further improved the above models in the following ways:
- Make use of position information in the attention model
- Add a layer to the attention model, which improves the accuracy

We include the pre-trained attention model as a tar.gz in this repo, which is last missing piece of pre-trained models needed in those two repos.
The users will need to decompress it. Also, the users need to follow the installation instruments in the two sub-folders, and download the pre-trained models and dictionaries..