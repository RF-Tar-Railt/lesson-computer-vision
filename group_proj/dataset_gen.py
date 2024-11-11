from pathlib import Path
import random
import xml.etree.ElementTree as ET
from PIL import Image
import json


fruit_path = Path(r"C:\Users\TR\Desktop\python\fruit1")

datasets_path = Path("./yolo_datasets")
images_path = datasets_path / "images"
labels_path = datasets_path / "labels"

img_train = images_path / "train"
img_train.mkdir(parents=True, exist_ok=True)
img_test = images_path / "test"
img_test.mkdir(parents=True, exist_ok=True)
img_val = images_path / "val"
img_val.mkdir(parents=True, exist_ok=True)

label_train = labels_path / "train"
label_train.mkdir(parents=True, exist_ok=True)
label_test = labels_path / "test"
label_test.mkdir(parents=True, exist_ok=True)
label_val = labels_path / "val"
label_val.mkdir(parents=True, exist_ok=True)


images = {}
image_specs = []

for file in fruit_path.joinpath("train").iterdir():
    if file.suffix != ".jpg":
        continue
    typ = file.stem.split("_")[0]
    if typ not in images:
        images[typ] = []
    if file.stem == "apple_0":
        image_specs.append(file)
    else:
        images[typ].append(file)


for file in fruit_path.joinpath("test").iterdir():
    if file.suffix != ".jpg":
        continue
    typ = file.stem.split("_")[0]
    if typ not in images:
        images[typ] = []
    images[typ].append(file)


trains = [*image_specs]
tests = []
vals = []

for name in images:
    random.shuffle(images[name])
    len_ = len(images[name])
    trains.extend(images[name][: int(len_ * 0.6)])
    tests.extend(images[name][int(len_ * 0.6) : int(len_ * 0.8)])
    vals.extend(images[name][int(len_ * 0.8) :])


def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h


classes = []

def convert_annotation(path: Path):
    img = Image.open(path)
    w = img.width
    h = img.height
    if (json_file := path.with_suffix(".json")).exists():
        with json_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
            res = []
            for obj in data["shapes"]:
                typ = obj["label"]
                if typ not in classes:
                    classes.append(typ)
                typ_id = classes.index(typ)
                b = (
                    min(obj["points"][0]),
                    min(obj["points"][1]),
                    max(obj["points"][0]),
                    max(obj["points"][1]),
                )
                bb = convert((w, h), b)
                res.append(f"{typ_id} {' '.join(str(a) for a in bb)}")
            return "\n".join(res)
    with path.with_suffix(".xml").open("r", encoding="utf-8") as f:
        tree = ET.parse(f)
        root = tree.getroot()
        res = []
        for obj in root.iter("object"):
            typ = obj.find("name").text
            if typ not in classes:
                classes.append(typ)
            typ_id = classes.index(typ)
            box = obj.find("bndbox")
            b = (
                float(box.find("xmin").text),
                float(box.find("xmax").text),
                float(box.find("ymin").text),
                float(box.find("ymax").text),
            )
            bb = convert((w, h), b)
            res.append(f"{typ_id} {' '.join(str(a) for a in bb)}")
        return "\n".join(res)



for i, file in enumerate(trains):
    label = label_train / f"{i}.txt"
    with label.open("w+", encoding="utf-8") as f:
        f.write(convert_annotation(file))
    with (img_train / f"{i}.jpg").open("wb+") as f:
        image = Image.open(file)
        image = image.convert("RGB")
        image.save(
            f,
            format="JPEG",
            quality=100,
        )


for i, file in enumerate(tests):
    label = label_test / f"{i}.txt"
    with label.open("w+", encoding="utf-8") as f:
        f.write(convert_annotation(file))
    with (img_test / f"{i}.jpg").open("wb+") as f:
        image = Image.open(file)
        image = image.convert("RGB")
        image.save(
            f,
            format="JPEG",
            quality=100,
        )


for i, file in enumerate(vals):
    label = label_val / f"{i}.txt"
    with label.open("w+", encoding="utf-8") as f:
        f.write(convert_annotation(file))
    with (img_val / f"{i}.jpg").open("wb+") as f:
        image = Image.open(file)
        image = image.convert("RGB")
        image.save(
            f,
            format="JPEG",
            quality=100,
        )

print(classes)

template = """\
path: C://Users/TR/Desktop/python/cvison/assignment1/cvision_workspace/group_assignment/yolo_datasets
train: images/train
val: images/val
test: images/test

"""

with (datasets_path / "fruit.yaml").open("w+", encoding="utf-8") as f:
    template += f"nc: {len(classes)}\n"
    template += "names:\n"
    for i, name in enumerate(classes):
        template += f"  {i}: {name}\n"
    f.write(template)
