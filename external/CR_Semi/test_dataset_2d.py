"""
python test_dataset_2d.py
python external\CR_Semi\test_dataset_2d.py
"""

from dataop.dataset_2d import FollicleDataset2D

images_dir = r"D:\Uni_Bamberg\8_Semester\Bachelor thesis\Annotating Borna dataset\follicle-annotation-models\nnunet_data\Dataset001_Follicles\imagesTr"

labels_dir = r"D:\Uni_Bamberg\8_Semester\Bachelor thesis\Annotating Borna dataset\follicle-annotation-models\nnunet_data\Dataset001_Follicles\labelsTr"

dataset = FollicleDataset2D(
    images_dir=images_dir,
    labels_dir=labels_dir
)

print("Number of cases:", len(dataset))

image, mask, case_id = dataset[0]

print("Case:", case_id)
print("Image shape:", image.shape)
print("Image dtype:", image.dtype)
print("Image min/max:", image.min().item(), image.max().item())

print("Mask shape:", mask.shape)
print("Mask dtype:", mask.dtype)
print("Mask unique values:", mask.unique())