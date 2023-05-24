# Generalization of vision pre-trained models for histopathology
This is the repository for generalization of vision pre-trained models for histopathology:

## Citing this work

If you use this code or the results for your research, please cite:

```
@article{sikaroudi2023generalization,
  title={Generalization of vision pre-trained models for histopathology},
  author={Sikaroudi, Milad and Hosseini, Maryam and Gonzalez, Ricardo and Rahnamayan, Shahryar and Tizhoosh, HR},
  journal={Scientific reports},
  volume={13},
  number={1},
  pages={6065},
  year={2023},
  publisher={Nature Publishing Group UK London}
}
```

## Getting Started

These instructions will guide you on how to execute the code on your local machine for development and testing purposes.

### Prerequisites

You need to have the following packages installed:
- Python 3.7+
- PyTorch 1.9.0+
- numpy
- pandas
- sklearn
- wandb
- torchvision
- PIL

You can install these packages using pip:

```
pip install torch torchvision numpy pandas sklearn wandb pillow
```

### Configuration

To run the program, you need to have a configuration file (`config.json`). Here is a sample configuration file that you can use:

```json
{
    "learning_rate": 0.0001,
    "momentum": 0.9,
    "epochs": 50,
    "batch_size": 64,
    "augmentation_in_training": false,
    "model": "kimianet",
    "pretrained": false,
    "kimianet_weight_path": "../kimianet_weights/KimiaNetPyTorchWeights.pth",
    "dataframe_root": "/isilon/datasets/camelyon17/",
    "trail_sites": ["center_0", "center_1", "center_2", "center_3", "center_4"],
    "holdout_trial_site": "center_0",
    "font_path": "/usr/share/fonts/type1/gsfonts/c059016l.pfb",
    "train_val_portions": [70,10]
}
```
Note: Please adjust the parameters according to your needs and availability of computational resources.

### Running the code

To run the code, simply execute the main Python script:

```
python main.py
```

The script will then start training the model according to the parameters specified in the `config.json` file.

The model weights will be saved in the current directory with a name specified by the parameters in the `config.json` file.

## Contributions

Your contributions are always welcome. If you find a bug or want to propose a new feature, feel free to open an issue or send a pull request.

## Contact

If you need to get in touch with the maintainer of this project, please contact me at msikaroudi@uwaterloo.ca.

## License

This project is licensed under MIT License.

