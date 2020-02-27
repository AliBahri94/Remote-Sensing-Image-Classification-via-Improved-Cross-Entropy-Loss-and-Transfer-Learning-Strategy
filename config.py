########## The required configurations for training and testing phase ##########

cfg = dict()

## Dataset Address
cfg["Dataset_Address"]= "~/data/dataset name"  ## if you use raedy dataset path, you can enter link address in this path ==> "link address" 


## Image Format
cfg["Format_Img"]= ".png"


## Data Augmentation
cfg["Rotation_Range"]= 22
cfg["Width_Shift_Range"]= 0.19
cfg["Height_Shift_Range"]= 0.18
cfg["Horizontal_Flip"]= True
cfg["Fill_Mode"]= "nearest"

## Input Shape
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
cfg["Load_Model_Path"]= "~/checkpoint/pretrained model.h5"


## The path to save models + log files
cfg["Save_dir"]= "~/checkpoint"

##################################### End #######################################



########## The required configurations for predicting phase ##########

## Dataset Address
cfg["Dataset_Address_Evaluate"]= "~/data/dataset name/valid"   ## Note: 1. Consider Valid part, 2. You can use ready dataset address


## Path for Load Model
cfg["Pretrained_Model_Path"]= "~/checkpoint/tarained model.h5"

## Batch Size
cfg["Eval_Batch_Size"]= 1

############################### End ##################################







