########## The required configurations for training and testing phase ##########

cfg = dict()

## Dataset Address
cfg["Dataset_Address"]= "path/to/data"


## Image Format
cfg["Format_Img"]= ".png"


## Data Augmentation
cfg["Rotation_Range"]= 22
cfg["Width_Shift_Range"]= 0.19
cfg["Height_Shift_Range"]= 0.18
cfg["Horizontal_Flip"]= True
cfg["Fill_Mode"]= "nearest"

## Input SHape
cfg["Input_Shape"]= (256, 256, 3)


## Image Size for Resizing
cfg["Target_Size"]= (256 , 256)


## Class Mode (categorical or binary)
cfg["Class_Mode"]= "categorical"


## Learning Rate ==> Proposed (0.001 - 0.0001)
cfg["lr"]= 0.001 #init


## Number of Epochs
cfg["Epochs"]= 500


## Batch Size
cfg["Train_Batch_Size"]= 32
cfg["Test_Batch_Size"]= 32


## Momentum
cfg["Momentum"]= 0.9


## Flag for Load Model
cfg["Load_Model_Flag"]= False


## Path for Load Model
cfg["Load_Model_Path"]= "path/to/checkpoint"


## The path to save models + log files
cfg["Save_dir"]= "path/to/checkpoint"

##################################### End #######################################



########## The required configurations for predicting phase ##########

## Dataset Address
cfg["Dataset_Address_Evaluate"]= "path/to/data/valid"   ## Note: Valid part is considered.


## Path for Load Model
cfg["Pretrained_Model_Path"]= "path/to/checkpoint"

## Batch Size
cfg["Eval_Batch_Size"]= 1

############################### End ##################################







