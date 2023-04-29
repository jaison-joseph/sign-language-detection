# Sign Language Detection

A brief description of what this project does.

## Table of Contents

- [Installation](#Installation)
- [Usage](#Usage)
- [Directory structure](#Directory%20Structure)
- [License](#License)

## Installation

Python <= 3.10.10 (required)

```bash
$ pip install -r requirements.txt
```

Or if you want to install the specific versions that were used in the project
```bash
$ pip install -r requirements_version_nums.txt
```

Also refer to `deps.txt` for further information on dependencies.

## Usage

Before you run the flask application, please refer to the comments at the top of `app.py`. <br>

Configure the following variables before running the application:

```python
SAMPLE_COUNT_ = 200                             # the number of samples recorded when the user requests to record samples
STORE_SIZE_ = 20                                # the maxmium number of samples that the store can hold before the user has to save them to record new ones
FOLDER_NAME_ = 'demo_data'                      # the name of the folder (will create if non-existent) to save the recorded samples to
MODEL_NAME_ = 'models/v5/a2z_v9_model.model'    # the location of the model to load, keep empty if you don't wish to load one
```

To run the Flask application. <br>
```bash
$ python app.py
```

To run the training program.
```bash
$ python train.py
```

## Directory Structure
* `data/`: all training and testing data
* `models/`: all versions of the trained SVM's
* `play/`:  toy/helper/tester code
* `static/`: CSS pages for the flask application
* `templates/`: HTML pages for the flask application
* `app.py`: the Flask application 
* `train.py`: the Flask application 
* `deps.txt`: notes and links to the dependencies of the project 
* `notes.txt`: notes on the training of the models (`models/v1` to `models/v5`)
* `requirements.txt`: project dependencies (for pip)
* `requirements_version_nums.txt`: project dependencies with version numbers used for this project (for pip)


## License

[MIT](https://choosealicense.com/licenses/mit/)