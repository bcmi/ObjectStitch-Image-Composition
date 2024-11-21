# ObjectStitch-Image-Composition
**This is an unofficial implementation of the paper ["ObjectStitch: Object Compositing with Diffusion Model"](https://openaccess.thecvf.com/content/CVPR2023/papers/Song_ObjectStitch_Object_Compositing_With_Diffusion_Model_CVPR_2023_paper.pdf), CVPR 2023.**

Following ObjectStitch, our implementation takes masked foregrounds as input and utilizes both class tokens and patch tokens as conditional embeddings. Since ObjectStitch does not release their training dataset, we train our models on a large-scale public dataset [Open-Images](https://storage.googleapis.com/openimages/web/index.html). 

ObjectStitch is very robust and adept at adjusting the pose/viewpoint of inserted foreground object according to the background. However, the details could be lost or altered for those complex or rare objects. 

For better detail preservation and controllability, you can refer to our [ControlCom](https://github.com/bcmi/ControlCom-Image-Composition) and [MureObjectStitch](https://github.com/bcmi/MureObjectStitch-Image-Composition). **ControlCom and MureObjectStitch have been integrated into our image composition toolbox libcom https://github.com/bcmi/libcom. Welcome to visit and try ＼(^▽^)／** 

**Note that in the provided foreground image, the foreground object should occupy the whole foreground image (see our example), otherwise the performance would be severely affected.**


## Introduction

Our implementation is based on [Paint-by-Example](https://github.com/Fantasy-Studio/Paint-by-Example), utilizing masked foreground images and employing all class and patch tokens from the foreground image as conditional embeddings. The model is trained using the same hyperparameters as Paint-by-Example. Foreground masks for training images are generated using [Segment-Anything](https://github.com/facebookresearch/segment-anything).

In total, our model is trained on approximately 1.8 million pairs of foreground and background images from both the train and validation sets of Open-Images. Training occurs over 40 epochs, utilizing 16 A100 GPUs with a batch size of 16 per GPU.

## Get Started

### 1.  Dependencies

  - torch==1.11.0
  - pytorch_lightning==1.8.1
  - install dependencies:
    ```bash
    cd ObjectStitch-Image-Composition
    pip install -r requirements.txt
    cd src/taming-transformers
    pip install -e .
    ```

### 2.  Download Models

  - Please download the following files to the ``checkpoints`` folder to create the following file tree:
    ```bash
    checkpoints/
    ├── ObjectStitch.pth
    └── openai-clip-vit-large-patch14
        ├── config.json
        ├── merges.txt
        ├── preprocessor_config.json
        ├── pytorch_model.bin
        ├── tokenizer_config.json
        ├── tokenizer.json
        └── vocab.json
    ```
  - **openai-clip-vit-large-patch14 ([Huggingface](https://huggingface.co/BCMIZB/Libcom_pretrained_models/blob/main/openai-clip-vit-large-patch14.zip) | [ModelScope](https://www.modelscope.cn/models/bcmizb/Libcom_pretrained_models/file/view/master/openai-clip-vit-large-patch14.zip))**.

  - **ObjectStitch.pth ([Huggingface](https://huggingface.co/BCMIZB/Libcom_pretrained_models/blob/main/ObjectStitch.pth) | [ModelScope](https://www.modelscope.cn/models/bcmizb/Libcom_pretrained_models/file/view/master/ObjectStitch.pth))**.
  

### 3. Inference on examples

  - To perform image composition using our model, you can use `scripts/inference.py`. For example,

    ```
    python scripts/inference.py \
    --outdir results \
    --testdir examples \
    --num_samples 3 \
    --sample_steps 50 \
    --gpu 0
    ```
    or simply run:
    ```
    sh test.sh
    ```
    These images under ``examples`` folder are obtained from [COCOEE](https://github.com/Fantasy-Studio/Paint-by-Example) dataset. 

### 4. Create composite images for your dataset

- Please refer to the [examples](./examples/) folder for data preparation:
  - keep the same filenames for each pair of data. 
  - either the ``mask_bbox`` folder or the ``bbox`` folder is sufficient.
 
### 5. Training Code

- **Download link**:
  - [Google Drive](https://drive.google.com/file/d/12Q0SHybm3mnd8d84wXEFVU2zRfjtp00K/view?usp=sharing)
  - [Baidu Netdisk](https://pan.baidu.com/s/1jKLt1yJfPMQExAzlgM3VLQ?pwd=b5ro)

**Notes**: certain sensitive information has been removed since the model training was conducted within a company. To start training, you'll need to prepare your own training data and make necessary modifications to the code according to your requirements.

## Visualization Results

We showcase several results generated by the released model on [FOSCom](https://github.com/bcmi/ControlCom-Image-Composition?tab=readme-ov-file#FOSCom-Dataset) dataset. In each example, we display the background image with a bounding box (yellow), the foreground image, and 5 randomly sampled images. 

<p align='center'>  
  <img src='./results/FOSCom_results.jpg'  width=95% />
</p>

We also provide the full results of 640 foreground-background pairs on FOSCom dataset, which can be downloaded from [Baidu Cloud](https://pan.baidu.com/s/1x-GQ0qzLFT2HOBvNDpcwsA?pwd=9g1p) (9g1p). Based on the results, you can quickly know the ability and limitation of ObjectStitch. 

## Other Resources
+ We summarize the papers and codes of generative image composition: [Awesome-Generative-Image-Composition](https://github.com/bcmi/Awesome-Generative-Image-Composition)
+ We summarize the papers and codes of image composition from all aspects: [Awesome-Image-Composition](https://github.com/bcmi/Awesome-Image-Composition)
+ We summarize all possible evaluation metrics to evaluate the quality of composite images:  [Composite-Image-Evaluation](https://github.com/bcmi/Composite-Image-Evaluation)
+ We write a comprehensive survey on image composition: [the latest version](https://arxiv.org/pdf/2106.14490.pdf)
