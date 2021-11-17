from fastai.data.external import untar_data, URLs

coco_path = untar_data(URLs.COCO_SAMPLE)
coco_path = str(coco_path) + "/train_sample"
print(coco_path)

