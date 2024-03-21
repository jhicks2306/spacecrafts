### Spacecraft Object Detection: DrivenData

In this project, I used DrivenData's [Pose Bowl: Spacecraft Detection and Pose Estimation Challenge](https://www.drivendata.org/competitions/group/competition-nasa-spacecraft/) as source data to practice using image detection models.

The object detection task involves drawing a bounding box around spacecrafts in a large dataset of images. See example below.

![Spacecraft](/archive/spacecraft-image1.png)

I built a tested a training loop locally and then ran it remotely on Kaggle's GPUs. In the end, I did not make a submission to the coding challenge. I found that full training cycles were taking too long, even on Kaggle's GPUs, and I wanted to move onto other projects. Still, this project was valuable practice using leading deep learning frameworks.

#### Skills and technologies used
- File handling (`sys`, `pathlib` for general file management and `PIL` for image files)
- Pre-processing and visualisation (`numpy`, `pandas` and `matplotlib`)
- Modelling (`torch`, `torchvision`, `pytorch-lightning`)
- Training (`kaggle` for training large datasets. This required managing CPU/ GPU usage using `cuda` in pytorch).
- Monitoring training process with `tensorboard` and testing submission files in a `docker` container.


