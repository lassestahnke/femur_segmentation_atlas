# femur_segmentation_atlas
Atlas based segmentation of the left femur and pelvis. Registration is done in a linear and non-linear fashion. 

The project consists of:
 - Manual segmentation of the femoral head and pelvis using seeded region growing in 3D Slicer
 - Registration of atlas images to target image using SITK
    - First linear registration (Similarity transform)
    - Then non-linear registration (B-Spline)
 - Segmenting target image based on the agreement of registered atlas segmentations using majority voting.
 
 Furthermore, this repository includes implementations to evaluate the segmentation results using Hausdorff distance and DSC. Notice that the function is specific to this project, were the labels of interest had the value 2 and 4. 
 
 In addition, a there is a Jupyter Notebook, were a pipeline for Obturator Foramen detection was implemented. The used network was too complex for the data at hand. Train with more data and resample images to have isotropic pixels when trying to do the detection on your own. 
 
 The repo is structured in a way that the [main.py file](main.py) starts defines the atlas images as a list of paths and also defines images that are supposed to be segmented. The files [segmentation](segmentation.py), [registration](registration.py), [transformation](transformation.py) and [assessment](assessment.py) implement their respective functions. For more details about the project see [project_report.pdf](project_report.pdf).
