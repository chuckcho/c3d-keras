import os
from os.path import join as join

# [DIR]
folder_of_videos = join("..", "videos")
folder_of_clips = join("..", "clips")
subset_of_clips = join("..", "clips", "subset")
clips_move_to = ("..", "clips", "frames_already_extracted")
path_to_annotation_file = join("..", "annotations.pkl")
path_to_annotators = join("..", "clips", "annotators")
annotation_obj_name = "annotations.pkl"
to_root_of_datasets = join("..", "..", "data_5h_video_sep_clips_scaled_002")
dataset_dir = join("..", "..", "data", "data_8h_001")
train_data_dir = join(dataset_dir, "train")
valid_data_dir = join(dataset_dir, "valid")
test_data_dir = join(dataset_dir, "test")
model_iteration = 'model_c3d_058'
models_dir = join("..", "models", model_iteration)
model_name = join(models_dir, model_iteration + '.h5')
history_name = join(models_dir, model_iteration + '_history.pkl')

# [ANNOTATORS]
annotators = [
  "Christine Dougherty",
  "Dylan Larson",
  "Michael Stavros",
  "Nick Scichilone",
  "Shasta Carlsen",
  "Test User",
  "Lee James"
  ]

# [FILETYPES]
videos_type = ".mp4"
clips_type = ".mp4"
img_type = ".jpg"

# [DATASET]
dataset_folders = ["train", "valid", "test"]
dataset_split_points = {"train": "clip", "valid": "clip", "test": "video"}
portions = {"train": 0.7, "valid": 0.15, "test": 0.15}
classes = {"0": "bad", "1": "good"}
classes_to_nums = {"bad":0, "good":1}
scale_heighth = 256
scale_width = -1
max_heighth = 256
max_width = 256
clip_subset_total_hrs_of_media = 0.25
sizes = [1280 * 720 * 3]
# [CLIP]
clip_length = 5
digits_in_clip_identifier = 3
img_type = ".jpg"

# [ANNOTATIONS]
default_name = "annotations.pkl"
fieldnames = ["filenames", "annotations"]

# in_dim = (128, 171)
in_dim = (480, 848)

# Parameters
train_params = {
        'in_dim': in_dim,
        'out_dim': (16, 112, 112),
        'batch_size': 16,
        'n_classes': 2,
        'n_channels': 3,
        'every_nth_frame':8,
        'shuffle': True,
        'augment': True}

valid_params = {
        'in_dim': in_dim,
        'out_dim': (16, 112, 112),
        'batch_size': 16,
        'n_classes': 2,
        'n_channels': 3,
        'every_nth_frame':8,
        'shuffle': True,
        'augment': False}

test_params = {
        'in_dim': in_dim,
        'out_dim': (16, 112, 112),
        'batch_size': 16,
        'n_classes': 2,
        'n_channels': 3,
        'every_nth_frame':8,
        'shuffle': True,
        'augment': False}

