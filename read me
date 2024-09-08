The main file to run the pipeline is Main_.ipynb.
In that you can give path of the test folder where input images are present.
Main_. file contains the intiation code to run the project.
The input image is passed to detectron 2 which is a object detection model, here slices are formed based on bbox and those slices are next stored into a data frame based on category as tehy under go categorising in main_pat_way file.Only gene are stored in a particular data frame and are passed to mmocr for text recognition.
Mmocr identifies the gene from slices.
detectron 2 contains retinanet model which is one of the object detection model present in the pipeline.
mmocr.py contains mmocr code which is one of the text detection model.
Cfg.py contains all the possible settings/configurations which can be changed.
Main_pat_way.py contains the code which splits the genes and relations seperatley by category id. These are predetermined by giving splitting input into 4 categories such as genes,activation , inhabit, none , it also contains postprocessing and it gives results in this format - img_id "file_name" "category_id" "bbox
It also contains gene center where relations code is present.
Body_interface is just used as refernece for Main_pat_way.

