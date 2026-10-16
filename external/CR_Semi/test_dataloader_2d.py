from torch.utils.data import DataLoader, random_split

from dataop.dataset_2d import FollicleDataset2D


images_dir = r"D:\Uni_Bamberg\8_Semester\Bachelor thesis\Annotating Borna dataset\follicle-annotation-models\nnunet_data\Dataset001_Follicles\imagesTr"

labels_dir = r"D:\Uni_Bamberg\8_Semester\Bachelor thesis\Annotating Borna dataset\follicle-annotation-models\nnunet_data\Dataset001_Follicles\labelsTr"


dataset = FollicleDataset2D(
    images_dir=images_dir,
    labels_dir=labels_dir
)


train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size


train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size]
)


train_loader = DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=2,
    shuffle=False
)


print("Train cases:", len(train_dataset))
print("Validation cases:", len(val_dataset))


images, masks, case_ids = next(iter(train_loader))

print("Batch image shape:", images.shape)
print("Batch mask shape:", masks.shape)
print("Case IDs:", case_ids)
print("Mask values:", masks.unique())