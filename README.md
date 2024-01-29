# UVLTrack

The official implementation of our AAAI 2024 paper [**Unifying Visual and Vision-Language Tracking via Contrastive Learning**](https://arxiv.org/abs/2401.11228)
![](arch.png)


## Brief Introduction
Single object tracking aims to locate the target object in a video sequence according to the state specified by different modal references, including the initial bounding box (BBOX), natural language (NL), or both (NL+BBOX). Different previous modality-specific trackers, we present a unified tracker called UVLTrack, which can simultaneously handle all three reference settings (BBOX, NL, NL+BBOX) with the same parameters, allowing wider appliation scenarios. The proposed UVLTrack enjoys several merits. First, we design a modality-unified feature extractor for joint visual and language feature learning and propose a multi-modal contrastive loss to align the visual and language features into a unified semantic space. Second, a modality-adaptive box head is proposed, which makes full use of the target reference to mine ever-changing scenario features dynamically from video contexts and distinguish the target in a contrastive way, enabling robust performance in different reference settings.

## Strong Performance
UVLTrack presents strong performance under different reference settings.
|        Reference Modality       |       NL       |       BBOX       |       NL+BBOX       |
|:--------------:|:---------:|:---------:|:---------:|
| Methods      | TNL2K (AUC / P)      | TNL2K(AUC / P)     | TNL2K(AUC / P)     |
| OSTrack-256  | -/- |54.3/-|-/-|
| OSTrack-384  | -/- |55.9/-|-/-|
| VLTTT |-/-|-/-|53.1/53.3|
| JointNLT  | 54.6/55.0 | -/- | 56.9/58.1 |
| UVLTrack-B   |   **55.7/57.2**    |   **62.7/65.4**    |   **63.1/66.7**    |
| UVLTrack-L   |   **58.2/60.9**    |   **64.8/68.8**    |   **64.9/69.3**    |

## Checkpoints
You can download the model weights and raw_result from [Google Drive](https://drive.google.com/drive/folders/1UZTrGcL3YlxvNpHi0wKsO_sKsTYuYnFo?usp=sharing) or [Baidu Netdisk](https://pan.baidu.com/s/1xSRgHHZKv8MyqMKwtT_JsA?pwd=8v0p).

|             Methods             |         UVLTrack-B       |       UVLTrack-L       |
|:-------------------------------:|:------------------------:|:------------------------:|
|       Reference Modality        |     TNL2K (AUC / P)      | TNL2K(AUC / P)     |
| NL      |   55.0/56.8  | 58.2/60.9 |
| BBOX    |   63.0/66.3  | 65.2/69.5 |
| NL+BBOX |   62.9/66.5  | 64.8/69.2 |

## Install the environment
Use the Anaconda
```
conda env create -f uvltrack_env.yaml
```
or
```
conda create -n uvltrack python=3.6
conda activate uvltrack
bash install.sh
```

## Data Preparation
Put the tracking datasets in ./data. It should look like:
   ```
   ${UVLTrack_ROOT}
    -- data
        -- lasot
            |-- airplane
            |-- basketball
            |-- bear
            ...
        -- lasotext
            |-- atv
            |-- badminton
            |-- cosplay
            ...
        -- got10k
            |-- test
            |-- train
            |-- val
        -- coco
            |-- annotations
            |-- train2017
        -- trackingnet
            |-- TRAIN_0
            |-- TRAIN_1
            ...
            |-- TRAIN_11
            |-- TEST
        -- otb99
            |-- OTB_query_test
            |-- OTB_query_train
            |-- OTB_videos
        -- refcocog
            |-- refcocog
            |-- train2014 # coco 2014
            |-- val2014 # coco 2014
        -- tnl2k
            |-- test
            |-- train
   ```

## Train UVLTrack
Download the pretrained [MAE](https://github.com/facebookresearch/mae) and [BERT](https://drive.google.com/drive/folders/1UZTrGcL3YlxvNpHi0wKsO_sKsTYuYnFo?usp=sharing), put it under ```<PROJECT_ROOT>/pretrain```.

Training with multiple GPUs using DDP.
```
# UVLTrack
sh scripts/train.sh uvltrack baseline_base
```

## Evaluation
Download the model weight from [Google Drive](https://drive.google.com/drive/folders/1UZTrGcL3YlxvNpHi0wKsO_sKsTYuYnFo?usp=sharing) or [Baidu Netdisk](https://pan.baidu.com/s/1xSRgHHZKv8MyqMKwtT_JsA?pwd=8v0p).

Put the downloaded weights on ```<PROJECT_ROOT>/checkpoints/train/uvltrack/baseline_base``` and ```<PROJECT_ROOT>/checkpoints/train/uvltrack/baseline_large``` correspondingly.

Notably, the modality of target reference (NL, BBOX or NLBBOX) is specified in config ```TEST.MODE```

```
# Testing
sh scripts/test.sh uvltrack baseline_base <dataset_name> <num_threads_per_gpu> <num_gpu>

# Evaluation
python tracking/analysis_results.py --tracker_name uvltrack --tracker_param baseline_base --dataset_name <dataset_name>_<reference_modality>_<EPOCH>

# Example
sh scripts/test.sh uvltrack baseline_base otb99 4 2
python tracking/analysis_results.py --tracker_name uvltrack --tracker_param baseline_base --dataset_name otb99_NL_300
```

## Run UVLTrack on your own video
Specify the target by bounding box or natural language, which should keep consistent with ```TEST.MODE``` in config.
```
sh scripts/demo.sh uvltrack baseline_base \
                   <input video path> \
                   <output video path> \
                   <language description of target> \
                   <initial bbox of target: x y w h>
```

## Test Speed
```
# UVLTrack-B: About 60 FPS on NVIDIA RTX 3090 GPU
python tracking/profile_model.py --script uvltrack --config baseline_base
# UVLTrack-L: About 34 FPS on NVIDIA RTX 3090 GPU
python tracking/profile_model.py --script uvltrack --config baseline_large
```

## Contact
For questions about our paper or code, please contact [Yinchao Ma](imyc@mail.ustc.edu.cn) or [Yuyang Tang](yuyangtang@mail.ustc.edu.cn)

## Acknowledgments
* Thanks for [JointNLT](https://github.com/lizhou-cs/JointNLT), [Mixformer](https://github.com/MCG-NJU/MixFormer) and [OSTrack](https://github.com/botaoye/OSTrack) Library, which helps us to quickly implement our ideas.

* We use the implementation of the ViT from the [Timm](https://github.com/huggingface/pytorch-image-models) repo and BERT from the [pytorch_pretrained_bert](https://github.com/Meelfy/pytorch_pretrained_BERT).

## Citation
If our work is useful for your research, please consider cite:
```
@misc{ma2024unifying,
      title={Unifying Visual and Vision-Language Tracking via Contrastive Learning}, 
      author={Yinchao Ma and Yuyang Tang and Wenfei Yang and Tianzhu Zhang and Jinpeng Zhang and Mengxue Kang},
      year={2024},
      eprint={2401.11228},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
