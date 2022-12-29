# femur_segmentation_atlas
Atlas based segmentation of the left femur and pelvis. Registration is done in a linear and non-linear fashion. 

The project consists of:
 - Manual segmentation of the femoral head and pelvis using seeded region growing in 3D Slicer
 - Registration of atlas images to target image using SITK
    - First linear registration
    - Then non-linear registration
 - Segmenting target image based on the agreement of registered atlas segmentations
