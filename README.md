## Demo for Visual Question Answering with BUTD

This an user friendly demo for visual question answering. It is essentially a pipeline that combines
[an image feature extractaion tool](https://github.com/peteanderson80/bottom-up-attention) and
[a fast attention implementation](https://github.com/hengyuan-hu/bottom-up-attention-vqa/). 
These two repos implement the BUTD system described in "Bottom-Up and
Top-Down Attention for Image Captioning and Visual Question Answering"
(https://arxiv.org/abs/1707.07998) and "Tips and Tricks for Visual
Question Answering: Learnings from the 2017 Challenge"
(https://arxiv.org/abs/1708.02711).

We further improved the above models in the following ways:
- Make use of position information in the attention model
- Add a layer to the attention model, which improves the accuracy

We include the pre-trained attention model as a tar.gz in this repo.
You need to decompress it. Also, you need to clone and install the tool
[here](https://github.com/peteanderson80/bottom-up-attention).