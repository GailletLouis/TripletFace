Triplet loss for facial recognition.

# Triplet Face

The repository contains code for the application of triplet loss training to the
task of facial recognition. This code has been produced for a lecture and is not
going to be maintained in any sort.

![TSNE_Latent](TSNE_Latent.png)

## Architecture

The proposed architecture is pretty simple and does not implement state of the
art performances. The chosen architecture is a fine tuning example of the
resnet18 CNN model. The model includes the freezed CNN part of resnet, and its
FC part has been replaced to be trained to output latent variables for the
facial image input.

The dataset needs to be formatted in the following form:
```
dataset/
| test/
| | 0/
| | | 00563.png
| | | 01567.png
| | | ...
| | 1/
| | | 00011.png
| | | 00153.png
| | | ...
| | ...
| train/
| | 0/
| | | 00001.png
| | | 00002.png
| | | ...
| | 1/
| | | 00001.png
| | | 00002.png
| | | ...
| | ...
| labels.csv        # id;label
```

## Install

Install all dependencies ( pip command may need sudo ):
```bash
cd TripletFace/
pip3 install -r requirements.txt
```

## Usage

For training:
```bash
usage: train.py [-h] -s DATASET_PATH -m MODEL_PATH [-i INPUT_SIZE]
                [-z LATENT_SIZE] [-b BATCH_SIZE] [-e EPOCHS]
                [-l LEARNING_RATE] [-w N_WORKERS] [-r N_SAMPLES]

optional arguments:
  -h, --help            show this help message and exit
  -s DATASET_PATH, --dataset_path DATASET_PATH
  -m MODEL_PATH, --model_path MODEL_PATH
  -i INPUT_SIZE, --input_size INPUT_SIZE
  -z LATENT_SIZE, --latent_size LATENT_SIZE
  -b BATCH_SIZE, --batch_size BATCH_SIZE
  -e EPOCHS, --epochs EPOCHS
  -l LEARNING_RATE, --learning_rate LEARNING_RATE
  -w N_WORKERS, --n_workers N_WORKERS
  -r N_SAMPLES, --n_samples N_SAMPLES
```

## References

* Resnet Paper: [Arxiv](https://arxiv.org/pdf/1512.03385.pdf)
* Triplet Loss Paper: [Arxiv](https://arxiv.org/pdf/1503.03832.pdf)
* TripletTorch Helper Module: [Github](https://github.com/TowardHumanizedInteraction/TripletTorch)

## Todo ( For the students )

**Deadline Decembre 13th 2019 at 12pm**

The students are asked to complete the following tasks:
* Fork the Project
* Improve the model by playing with Hyperparameters and by changing the Architecture ( may not use resnet )
* JIT compile the model ( see [Documentation](https://pytorch.org/docs/stable/jit.html#torch.jit.trace) )
* Add script to generate Centroids and Thesholds using few face images from one person
* Generate those for each of the student included in the dataset
* Add inference script in order to use the final model
* Change README.md in order to include the student choices explained and a table containing the Centroids and Thesholds for each student of the dataset with a vizualisation ( See the one above )
* Send the github link by mail

## Done :
* Fork the project
* Creating a first model 'model' with the default settings except epochs=3 (runtime = 1h30)
* Creating a second model 'model_2' with the default settings except epochs=5
(runtime = 2h30)
* Creating a jitcompile.py for both models
* preparing a third model with new parameters :
  * batch_size = 64 instead of 32 should be better but will take more time to train
  * Learning rate : one of the most important hyperparameter : which is the "speed at which the model learns". learning_rate = 5*1e-4 instead of 1e-3 The bigger is the learning rate, the faster the model is trained, but it will not learn at its best. But if the learning rate is too small, the training may get stucked. So I halved it.
  * let the n_workers to 4 : it should use most of the gpu. Found out that the recommended of workers should be 4 * the number of gpu we have. I assume colab gives only 1 gpu.

## What I learned but could not do :
* Resnet is a neural network already trained for image classification and we want to specialize it with the triplet loss method. We could have change resnet for another pytorch neural network at the line  :
  self.resnet    = models.resnet18( pretrained = True )
but i did not know which one to choose and why
* Centroid is the average position of all profilenpictures on the graph in our project.
* Threshold is the limite of the zone which determine whether or not a photo is similar to a group of photos representing one person