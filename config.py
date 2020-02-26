########## The required configurations for training phase ##########

cfg = dict()

## Dataset Address
cfg["Dataset_Address"]= "/path/to/data"


## Image Format
cfg["Format_Img"]= ".png"


## Data Augmentation
cfg["Rotation_Range"]= 22
cfg["Width_Shift_Range"]= 0.19
cfg["Height_Shift_Range"]= 0.18
cfg["Horizontal_Flip"]= True
cfg["Fill_Mode"]= "nearest"


## Image Size for Resizing
cfg["Target_Size"]= (255 , 255)


## Class Mode (categorical or binary)
cfg["Class_Mode"]= "categorical"


## Learning Rate ==> Proposed (0.001 - 0.0001)
cfg["lr"]= 0.001 #init


## Number of Epochs
cfg["Epochs"]= 500


## Batch Size
cfg["Train_Batch_Size"]= 32


## Momentum
cfg["Momentum"]= 0.9


## The path to save models + log files
cfg["Save_dir"]= "path/to/checkpoint"

############################### End ###############################


