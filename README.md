# bheEASyMesh
Tool for creating simple VTK unstructured grid meshes (3D prism elements) with BHEs (line elements) for use in OpenGeoSys 6 HeatTransportBHE simulations. Besides, the extraction of surface meshes and the application of temperature ICs is possible. The meshing is based on the Bhe_Meshing tool by Philip Hein (Shao et al., 2016) and uses GMSH (Geuzaine et al., 2009).<br />
Meshes contain:<br />
    &emsp;- 3D prism mesh with horizontal element layers<br />
    &emsp;- A selected number of BHEs, which are represented as line elements<br />
    &emsp;- Diersch's ideal node distance around BHE-elements (Diersch et al., 2011)<br />
    &emsp;- A chosen number of optional horizontal refinement boxes in the mesh<br />
    &emsp;- Extraction of surfacemeshes to define BC's or IC's in OpenGeoSys<br />
    &emsp;- An optional temperature IC as depth dependent profile or fixed temperature which can be applied to the mesh and if needed to surfacemeshes<br />


There are two modes in the Software:<br />
=>"simple" Meshes:<br />
    &emsp;- Simple meshes with two different material groups to be used for geological homogeneous models with an area with groundwater flow <br />
    &emsp;- the software automatically adjusts the vertical layers over the user-defined default element height<br />
    &emsp;- refined layers in the transition zones (BHE top/bottom, aquifer top/bottom) by user-defined number of refined layers and a separate element height for refined areas<br />

=>"layered" Meshes:<br />
    &emsp;- manual definition of any number of layers<br />
    &emsp;- number of elements, height of the elements and material ID can be freely selected for each layer <br />

For more instructions, see Meshing_example.ipynb, where there are tutorials for both meshing modes.<br />


====================================<br />
Diersch et al. (2011): Finite element modeling of borehole heat exchanger systems Part 2.Numerical simulation. In: Computers & Geosciences 37, pages 1136-1147.<br />

Geuzaine C et al. (2009) Gmsh: A 3-D finite element mesh generator with built-in pre- and post-processing facilities. International Journal for Numerical Methods in Engineering 79, pages 1309-1331.

Shao et al. (2016): Geoenergy Modeling II Shallow Geothermal Systems. Springer.