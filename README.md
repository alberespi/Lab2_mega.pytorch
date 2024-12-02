# Lab 2 Session 1 - MEGA for Video Object Detection

Authors: [Alberto Espinosa Pérez](https://github.com/alberespi), [Sergio del Pino Garvi](https://github.com/sergiodelpi)

Original authors: [Yihong Chen](https://scalsol.github.io), [Yue Cao](http://yue-cao.me), [Han Hu](https://ancientmooner.github.io/), [Liwei Wang](http://www.liweiwang-pku.com/).

Source repo: https://github.com/Scalsol/mega.pytorch

The execution of `Lab session 1` can be done by following the steps below:

## Steps:
1. ```
    git clone https://github.com/alberespi/Lab2_mega.pytorch.git
    cd Lab2_mega.pytorch/
    git checkout session1
    ```
2. Follow the steps in the `Install.md` or copy the following commands. Make sure to edit the `$PWD` to the desire working path:
   
    ```bash
    conda create --name MEGA -y python=3.7
    source activate MEGA

    conda install ipython pip
    pip install ninja yacs cython matplotlib tqdm opencv-python scipy
    conda install pytorch=1.2.0 torchvision=0.4.0 cudatoolkit=10.0 -c pytorch

    export INSTALL_DIR=$PWD

    cd $INSTALL_DIR
    git clone https://github.com/cocodataset/cocoapi.git
    cd cocoapi/PythonAPI
    python setup.py build_ext install
    
    cd $INSTALL_DIR
    git clone https://github.com/mcordts/cityscapesScripts.git
    cd cityscapesScripts/
    python setup.py build_ext install
    
    cd $INSTALL_DIR
    git clone https://github.com/NVIDIA/apex.git
    cd apex
    git checkout e3794f422628d453b036f69de476bf16a0a838ac
    python setup.py build_ext install
    
    cd $INSTALL_DIR
    git clone https://github.com/Scalsol/mega.pytorch.git
    cd mega.pytorch
    python setup.py build develop

    pip install 'pillow<7.0.0'

    unset INSTALL_DIR
    ```
    
3. In the `mega.pytorch\demo\predictor.py`, modify the following code to avoid the error in opencv:
   
    Previous:
    ```
    cv2.putText(
                image, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )
    ```
    After:
    ```
    cv2.putText(
                image, s, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )
    ```
    
4. Go to the `Main Results` section and download `single frame baseline`, I use `single-frame baseline` and `MEGA` model checkpoints (backbone ResNet-101). **Place model `R_101.pth` and `MEGA_R_101.pth` in the `mega.pytorch` folder.**

5. Using `demo/README.md` instructions of the `Inference on a folder` mode, run the demo code using both BASE and MEGA approaches. We are using some test frames stored in `datasets/ILSVRC2015/DATA/DET/image_folder`:
   
    BASE:
    ```bash
        python demo/demo.py base configs/vid_R_101_C4_1x.yaml R_101.pth \
            --visualize-path datasets/ILSVRC2015/Data/image_folder --suffix ".JPEG"\
            --output-folder visualization
    ```
    
    MEGA:
    ```bash
        python demo/demo.py mega configs/MEGA/vid_R_101_C4_MEGA_1x.yaml MEGA_R_101.pth \
            --visualize-path datasets/ILSVRC2015/Data/image_folder \
            --suffix ".JPEG" --output-folder visualization_mega
    ```

    To generate a video from the frames add the flag `--output-video` at the end of the command.

    Outputs are available in `visualization` and `visualization_mega` for BASE and MEGA approaches respectively.

## Main Results

Pretrained models are now available at [Baidu](https://pan.baidu.com/s/1qjIAD3ohaJO8EF1mZ4nLEg) (code: neck) and Google Drive.

Model | Backbone | AP50 | AP (fast) | AP (med) | AP (slow) | Link
:---: | :---: | :---: | :---: | :---: | :---: |:---:
single frame baseline | ResNet-101 | 76.7 | 52.3 | 74.1 | 84.9 | [Google](https://drive.google.com/file/d/1W17f9GC60rHU47lUeOEfU--Ra-LTw3Tq/view?usp=sharing)
DFF | ResNet-101 | 75.0 | 48.3 | 73.5 | 84.5 | [Google](https://drive.google.com/file/d/1Dn_RQRlA7z2XkRRS4XERUW_UH9jlNvMo/view?usp=sharing)
FGFA | ResNet-101 | 78.0 | 55.3 | 76.9 | 85.6 | [Google](https://drive.google.com/file/d/1yVgy7_ff1xVD1SooqbcK-OzKMgPpUcg4/view?usp=sharing)
RDN-base | ResNet-101 | 81.1 | 60.2 | 79.4 | 87.7 | [Google](https://drive.google.com/file/d/1jM5LqlVtCGjKH-MocTCjzFIVjqCyng8M/view?usp=sharing)
RDN | ResNet-101 | 81.7 | 59.5 | 80.0 | 89.0| [Google](https://drive.google.com/file/d/1FgoOwj-GFAMVn2hkSFKnxn5fKWPSxlUF/view?usp=sharing)
**MEGA** | ResNet-101 | 82.9 | 62.7| 81.6 | 89.4 | [Google](https://drive.google.com/file/d/1ZnAdFafF1vW9Lnpw-RPF1AD_csw61lBY/view?usp=sharing)

Model | Backbone | AP50 | AP (fast) | AP (med) | AP (slow) | Link
:---: | :---: | :---: | :---: | :---: | :---: |:---:
single frame baseline | ResNet-50 | 71.8 | 47.2 | 69.2 | 80.6| [Google](https://drive.google.com/file/d/1i39MwpP46x61eHLkRXMzcKhpeKZhkgA6/view?usp=sharing)
DFF | ResNet-50 | 70.4 | 43.6 | 68.9 | 80.8 | [Google](https://drive.google.com/file/d/1wl9Sheg46ecJOWzl1Uy4BWaCDRtSt51_/view?usp=sharing)
FGFA | ResNet-50 | 74.3 | 50.6 | 72.3 | 84.0|  [Google](https://drive.google.com/file/d/1nJ6CbUG_wW_gvMs193b7f0c1QLnXqAzO/view?usp=sharing)
RDN-base | ResNet-50 | 76.7 | 53.8 | 74.8 | 85.4 | [Google](https://drive.google.com/file/d/10k70lzSrxXiLWYx8tmX3RNuOQ2x1X0k8/view?usp=sharing)
**MEGA** | ResNet-50 | 77.3 | 56.5 | 75.7 | 85.2 | [Google](https://drive.google.com/file/d/1EZzpBuCfI75bsd_gxK1495tXlh0K_34H/view?usp=sharing)

**Note**: The performance of ResNet-50 backbone are not so stable.

**Note**: The motion-IoU specific AP evaluation code is a bit different from the original implementation in FGFA. I think the original implementation is really weird so I modify it. So the results may not be directly comparable with the results provided in FGFA and other methods that use MXNet version evaluation code. But we could tell which method is relatively better under the same evaluation protocol.
