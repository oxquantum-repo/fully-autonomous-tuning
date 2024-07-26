# Bias Triangle Detection

The notebook Bias_Triangle_Detection_example exemplifies the use of the bias triangle detection framework.
The approach broadly consists of 

1. Initial image preprocessing (as needed) & binary thresholding *
2. Initial contour detection
3. Contour approximation by minimum-edge polygon & maximum area overlap (using the RDP)
4. Polygon/Triangle feature extraction
    4.1 Extraction of the triangle base - requires user input of rough triangle direction ('top', 'bottom', 'right', 'left') & angles
    4.2 Extraction of the rough area between the base & first excited state 
    
Step 4 aims to facilitate the identification of Pauli-Spin-Blockade (PSB) & the readout spot (for qubit identification).     
    
The module ``im_utils.py`` contains basic image processing functions, the module ``btriangle_detection.py`` contains the detection/geometric approximation functionality and ``btriangle_properties.py`` contains the feature extraction functionality.

As per the notebook, the detection framework can be applied to either a single zoomed-in triangle or a wide shot image featuring many triangles - in the case of the latter, some default params may need to be adapted.

*If the shape at hand has a distinct gap between the base line & rest of the triangle, apply the thresholding method "noisy binary" (or "noisy_triangle"), optionally with Gaussian filtering. For data with extreme cases (such as from a FinFet device), further consider the sub_module with "allow_MET = True" in ``btriangle_detection.py``.
