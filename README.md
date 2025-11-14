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
--annotation-file./data/annotation.txt \
--histogram-file ./data/image_size_hist.png \
--class-ratio-file ./data/class_ratio.png \
--balance-classes
```
<img width="500" alt="class_ratio" src="https://github.com/user-attachments/assets/803c0341-bdd7-43fa-9569-c195a82f6434" /><img width="500" alt="image_size_hist" src="https://github.com/user-attachments/assets/04459a93-a08f-42f5-ab4e-196853145af0" />
