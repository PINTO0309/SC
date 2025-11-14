# SC
Sitting Classification.

- AVA Actions Download (v2.2) - CC BY 4.0 license
  https://research.google.com/ava/download.html

```bash
uv run python 01_download_ava_videos.py
```
```bash
uv run python 03_renumber_files.py
uv run python 04_data_prep.py

uv run python 04_data_prep.py \
--timestamp-stride 1 \
--image-dir /home/b920405/git/SC/data/images \
--annotation-file /home/b920405/git/SC/data/annotation.txt \
--histogram-file /home/b920405/git/SC/data/image_size_hist.png \
--class-ratio-file /home/b920405/git/SC/data/class_ratio.png \
--balance-classes
```
