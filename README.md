# bheEASyMesh
Tool for  building simple VTK unstructured grid meshes (3D Prism) with BHEs, for extracting surface meshes and for appling temperature IC's to be used in OpenGeoSys 6 HeatTransportBHE simulations. Meshing is based on Philip Heins Bhe_Meshing Tool and uses GMSH.

Meshes contain:
    - one BHE (type 2U) 
    - Diersch ideal node distance around BHE-elements
    - horizontal refinement box around the bhe
    - one aquifer (elements with different materialgroup)
    - refined Layers in the transition zones (BHE top/bottom, aquifer top/bottom)
  
For more instructions, see Meshing_example.ipynb.
