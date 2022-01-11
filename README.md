# Image-Text-Nontext-Classifier

A text/non-text image classifier. Part of software course project.

Root repo: https://github.com/KSkun/Image-Text-Nontext-Classifier-Service

## Features

### Neural Network

The classifier uses a trained VGG-16 neural network to classify images that contains text or not. The last
full-connected layer is replaced with a 2-output one. The network was trained with *TextDis benchmark* with parameters
shown in [training script](src/net/net_train.py). Training was completed
at [Google Colab](https://colab.research.google.com/).

As benchmark shows, the classifier has a F1 value of 0.859 at test dataset. Test dataset is a random subset of *TextDis
benchmark*, contains 2000 text images and 2000 non-text images, which are not contained in training dataset.

### Communication

The classifier uses [redis stream](https://redis.io/topics/streams-intro) to receive classification commands. It
registers itself to group `classifier` of stream `classfy_cmd`, and trys to fetch a command from the stream.

Command pattern:

- `op` field: operation, one of `init`, `one`, `many`
- `task_id` field: task ObjectId in MongoDB
- `image` field:
    - for `one` command: a JSON-encoded string like `{"id": "image ObjectId in MongoDB", "file": "image filename"}`
    - for `many` command: a JSON-encoded string of an array of image objects like the one in `one` command

The image file should be stored at `./workdir/tmp/<task ObjectId>/<image filename>`.

To cooperate with backend, after classification of each command, the classifier updates `class` field of each image
document in MongoDB.

### Tests

Unit tests are at `./test`. Includes function tests of the neural network.

## Configuration

### Local Startup

An example of working environment is in `./workdir`.

Config files store in `./workdir/config`, an example can be found as `default.json`.

Configuration steps:

1. Run `pip install -r requirements.txt`.
2. Set environment variable `CONFIG_FILE` to your config filename, if not set, it's `default.json`.
3. Create symlinks from image tmp, text, non-text directory to `./workdir`
4. Inside your config file, make sure the requirements below:
    1. `image_tmp_dir`, `image_text_dir` and `image_nontext_dir` set to your symlinks as step 2.
    2. `image_url` is your static resource site prefix.
    3. `mongo_xxx` is your global MongoDB settings.
    4. `redis_xxx` is your global Redis settings.
5. Create a redis stream `classify_cmd` and a consumer group `classifier` in your redis database.

   You can use an *init command* as `XADD classify_cmd * op init` to create the stream.
6. Run `python ../src/main.py` with working directory `./workdir`.

### Docker

See [docker configuration repo](https://github.com/KSkun/Image-Text-Nontext-Classifier-Service).