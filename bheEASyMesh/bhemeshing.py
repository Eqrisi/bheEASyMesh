import gmsh
import numpy as np
from vtk import *
from vtk.util import numpy_support
from scipy import interpolate


class BHEMesh():
    """
    Class for building VTK unstructured grid meshes with BHEs and for extracting surface meshes

    Meshes contain:
    - one BHE (type 2U) 
    - Diersch ideal node distance around BHE-elements
    - horizontal refinement box around the bhe
    - one aquifer (elements with different materialgroup)
    - refined Layers in the transition zones (BHE top/bottom, aquifer top/bottom)
    """


    def __init__(self, prefix,  meshInput, mode):
        self.prefix = prefix
        self.geom = {}
        self.BHEs = {}
        self.layers = {}
        self.cnt_mat_groups = 0 
        self.add_points ={}
        self.faces = {}
        self.use_temp_IC = False
        max_depth = 0
        for i in meshInput['BHEs']:
            self.BHEs[str(i)] = {}
            self.BHEs[str(i)]['bhe_number'] = i
            self.BHEs[str(i)]['bhe_x'] = meshInput['BHEs'][i]['Position'][0]
            self.BHEs[str(i)]['bhe_y'] = meshInput['BHEs'][i]['Position'][1]
            self.BHEs[str(i)]['bhe_top'] = meshInput['BHEs'][i]['Topend']
            self.BHEs[str(i)]['bhe_bottom'] = meshInput['BHEs'][i]['Topend'] - meshInput['BHEs'][i]['Length']
            self.BHEs[str(i)]['bhe_radius'] = meshInput['BHEs'][i]['r_b']
            self.BHEs[str(i)]['bhe_length'] = meshInput['BHEs'][i]['Length']
            if  meshInput['BHEs'][i]['Length'] + abs(meshInput['BHEs'][i]['Topend']) > max_depth:
                max_depth = meshInput['BHEs'][i]['Length'] + abs(meshInput['BHEs'][i]['Topend'])
        self.geom['width'] = meshInput['WIDTH']
        self.geom['length'] = meshInput['LENGTH']
        if 'BOXES' in meshInput:
            self.geom['box'] = meshInput['BOXES']
        self.geom['elem_size'] = meshInput['ELEM_SIZE']
        if 'ADD_POINTS' in meshInput:
            for i in range(len(meshInput['ADD_POINTS'])):
                this_point = {}
                this_point['x'] = meshInput['ADD_POINTS'][i][0]
                this_point['y'] = meshInput['ADD_POINTS'][i][1]
                this_point['delta'] = meshInput['ADD_POINTS'][i][2]
                self.add_points[str(len(self.add_points))] = this_point
        if mode == 'simple' or mode == 'Simple':
            self.geom['depth'] = max_depth + meshInput['zExt']
            self.geom['dz'] = meshInput['dzLAYER']
            self.geom['dz_ref'] = meshInput['dzREF']
            self.geom['n_ref'] = meshInput['nREF']
            self.geom['t_aqf'] = meshInput['t_aqf']
            self.geom['z_aqf'] = meshInput['z_aqf'] + abs(meshInput['inflow_max_z'])
            self.create_layer_structure()
        elif mode == 'layered' or mode == 'Layered':
            for i in range(len(meshInput['LAYERING'])):
                self.layers[str(i)] = {}
                self.layers[str(i)]['mat_group'] = meshInput['LAYERING'][i][0]
                self.layers[str(i)]['n_elems'] = meshInput['LAYERING'][i][1]
                self.layers[str(i)]['elem_thickness'] = meshInput['LAYERING'][i][2] 
        else:
            raise KeyError("'" + mode + "' is not a known mode, please choose 'layered' or 'simple' as mode")     
        self.write_GMSH_mesh()
        self.extrude_mesh()
        self.compute_BHE_elements()
        self.node_reordering()


    def create_layer_structure(self):
        dz = self.geom['dz']
        n_ref = self.geom['n_ref']
        dz_ref = self.geom['dz_ref']
        depth = self.geom['depth'] 
        z_aqf = self.geom['z_aqf']
        bheLen = self.BHEs['0']['bhe_length']
        t_aqf = self.geom['t_aqf']
        if z_aqf + t_aqf > depth:
            Layer_endings = {'topend': abs(self.BHEs['0']['bhe_top']),
                            'ref_start-bhebottom': bheLen + abs(self.BHEs['0']['bhe_top']) - (n_ref * dz_ref),
                            'bhebottom': bheLen + abs(self.BHEs['0']['bhe_top']),
                            'ref_end-bhebottom': bheLen + abs(self.BHEs['0']['bhe_top']) + (n_ref * dz_ref),
                            'ref_start_aqf_top': z_aqf - (n_ref * dz_ref),
                            'aqf_top': z_aqf,
                            'ref_end-aqf_top': z_aqf + (n_ref * dz_ref),
                            'ref_start-aqf_bottom': depth - (n_ref * dz_ref),
                            'aqf_bottom': depth}
        elif z_aqf + t_aqf + dz_ref * n_ref > depth:
            Layer_endings = {'topend': abs(self.BHEs['0']['bhe_top']),
                            'ref_start-bhebottom': bheLen + abs(self.BHEs['0']['bhe_top']) - (n_ref * dz_ref),
                            'bhebottom': bheLen + abs(self.BHEs['0']['bhe_top']),
                            'ref_end-bhebottom': bheLen + abs(self.BHEs['0']['bhe_top']) + (n_ref * dz_ref),
                            'ref_start_aqf_top': z_aqf - (n_ref * dz_ref),
                            'aqf_top': z_aqf,
                            'ref_end-aqf_top': z_aqf + (n_ref * dz_ref),
                            'ref_start-aqf_bottom': depth - (n_ref * dz_ref),
                            'aqf_bottom': z_aqf + t_aqf,
                            'bottom': depth}
        elif t_aqf == 0:
            Layer_endings = {'topend': abs(self.BHEs['0']['bhe_top']),
                            'ref_start-bhebottom': bheLen + abs(self.BHEs['0']['bhe_top']) - (n_ref * dz_ref),
                            'bhebottom': bheLen + abs(self.BHEs['0']['bhe_top']),
                            'ref_end-bhebottom': bheLen + abs(self.BHEs['0']['bhe_top']) + (n_ref * dz_ref),
                            'bottom': depth}
        else:           
            Layer_endings = {'topend': abs(self.BHEs['0']['bhe_top']),
                            'ref_start-bhebottom': bheLen + abs(self.BHEs['0']['bhe_top']) - (n_ref * dz_ref),
                            'bhebottom': bheLen + abs(self.BHEs['0']['bhe_top']),
                            'ref_end-bhebottom': bheLen + abs(self.BHEs['0']['bhe_top']) + (n_ref * dz_ref),
                            'ref_start-aqf_top': z_aqf - (n_ref * dz_ref),
                            'aqf_top': z_aqf,
                            'ref_end-aqf_top': z_aqf + (n_ref * dz_ref),
                            'ref_start-aqf_bottom': z_aqf + t_aqf - (n_ref * dz_ref),
                            'aqf_bottom': z_aqf + t_aqf,
                            'ref_end-aqf_bottom': z_aqf + t_aqf + (n_ref * dz_ref),
                            'bottom': depth}
        Layer_endings = {name: val for name, val in sorted(Layer_endings.items(), key=lambda item: item[1])}
        medium_idx = 0
        sum_z = 0
        ref = False
        ref_key = ""
        for var in Layer_endings:
            if 'ref_end' in var and var.split('-')[-1] != ref_key:
                continue
            if 'ref_start' in var and ref == True:
                ref_key = var.split('-')[-1]
                continue
            if Layer_endings[var] < 0:
                if 'ref_start' in var:
                    ref = True
                    ref_key = var.split('-')[-1]
                    continue
                else:    
                    Layer_endings[var] = 0      
            if ref == True:
                if Layer_endings[var]-sum_z >= n_ref * dz_ref:
                    thisLayer = {}
                    thisLayer['mat_group'] = medium_idx
                    thisLayer['n_elems'] = n_ref
                    thisLayer['elem_thickness'] = dz_ref
                    self.layers[str(len(self.layers))] = thisLayer
                    sum_z = sum_z + n_ref * dz_ref
                else:
                    n, rest = divmod(Layer_endings[var] - sum_z, dz_ref)
                    rest = Layer_endings[var] - (sum_z + n * dz_ref)
                    if not n == 0: 
                        thisLayer = {}
                        thisLayer['mat_group'] = medium_idx
                        thisLayer['n_elems'] = n
                        thisLayer['elem_thickness'] = dz_ref
                        self.layers[str(len(self.layers))] = thisLayer
                        sum_z = sum_z + n * dz_ref
                    if not rest == 0:
                        thisLayer = {}
                        thisLayer['mat_group'] = medium_idx
                        thisLayer['n_elems'] = 1
                        if rest < 1e-3 and n > 1:
                            self.layers[str(len(self.layers)-1)]['n_elems'] = n - 1    
                            thisLayer['elem_thickness'] = dz_ref + rest
                            self.layers[str(len(self.layers))] = thisLayer
                        elif rest < 1e-3 and n == 1: 
                            thisLayer['elem_thickness'] = dz_ref + rest
                            self.layers[str(len(self.layers)-1)] = thisLayer
                        elif rest >= 1e-5:
                            thisLayer['elem_thickness'] = rest
                            self.layers[str(len(self.layers))] = thisLayer
                        sum_z = sum_z + rest
                if 'ref_end' in var:
                    ref = False
            if 'ref_start' in var:
                ref = True
                ref_key = var.split('-')[-1]
            n, rest = divmod(Layer_endings[var] - sum_z, dz)
            rest = Layer_endings[var] - (sum_z + n * dz) ## rest has to be recalculated due to inaccuracies of divmod
            if not n == 0: 
                thisLayer = {}
                thisLayer['mat_group'] = medium_idx
                thisLayer['n_elems'] = n
                thisLayer['elem_thickness'] = dz
                self.layers[str(len(self.layers))] = thisLayer
                sum_z = sum_z + n * dz
            if not rest == 0:
                thisLayer = {}
                thisLayer['mat_group'] = medium_idx
                thisLayer['n_elems'] = 1
                if rest < 1e-3 and n > 1:
                    self.layers[str(len(self.layers)-1)]['n_elems'] = n - 1
                    thisLayer['elem_thickness'] = dz + rest
                    self.layers[str(len(self.layers))] = thisLayer
                elif rest < 1e-3 and n == 1: 
                    thisLayer['elem_thickness'] = dz + rest
                    self.layers[str(len(self.layers)-1)] = thisLayer
                elif rest >= 1e-5:
                    thisLayer['elem_thickness'] = rest
                    self.layers[str(len(self.layers))] = thisLayer
                sum_z = sum_z + rest
            if var == 'aqf_top': medium_idx = 1
            if var == 'aqf_bottom': medium_idx = 0 
            if n < 0: raise ValueError("Layer with n < 0 elements was calculated for layerending '" + var + "'!")


    def write_GMSH_mesh(self):
        n_BHEs = len(self.BHEs)
        n_add_points = len(self.add_points)
        cnt_pnt = 9
        point_idx = 0
        line_idx = 0
        lines_list = []
        gmsh.initialize()
        gmsh.model.add(self.prefix)
        point_idx +=1
        gmsh.model.geo.add_point(-self.geom["width"]/2, 0, 0, self.geom["elem_size"], point_idx)
        point_idx +=1
        gmsh.model.geo.add_point(self.geom["width"]/2, 0, 0, self.geom["elem_size"], point_idx)
        point_idx +=1
        gmsh.model.geo.add_point(self.geom["width"]/2, self.geom["length"], 0, self.geom["elem_size"], point_idx)
        point_idx +=1
        gmsh.model.geo.add_point(-self.geom["width"]/2, self.geom["length"], 0, self.geom["elem_size"], point_idx)
        line_idx += 1
        gmsh.model.geo.add_line(line_idx, point_idx - 2, point_idx - 3)
        lines_list.append(line_idx)
        line_idx += 1
        gmsh.model.geo.add_line(line_idx, point_idx - 1, point_idx - 2)
        lines_list.append(line_idx)
        line_idx += 1
        gmsh.model.geo.add_line(line_idx, point_idx, point_idx - 1)
        lines_list.append(line_idx)
        line_idx += 1
        gmsh.model.geo.add_line(line_idx, point_idx - 3, point_idx)
        lines_list.append(line_idx)
        gmsh.model.geo.addCurveLoop(lines_list, 1)
        gmsh.model.geo.addPlaneSurface([1], 1)
        if 'box' in self.geom:
            for i in self.geom['box']:
                if self.geom['box'][i]['Origin'][0] <= -0.5*self.geom['width'] or self.geom['box'][i]['Origin'][0] >= 0.5*self.geom['width']:
                    raise ValueError ("x-coordinate of box " + str(i) + " origin is outside the mesh")
                elif self.geom['box'][i]['Origin'][0] + self.geom['box'][i]['Width'] >= 0.5*self.geom['width'] or self.geom['box'][i]['Origin'][0] + self.geom['box'][i]['Width'] <= -0.5*self.geom['width']: 
                    raise ValueError ("x-Widht of box " + str(i) + " is outside the mesh")
                elif self.geom['box'][i]['Origin'][1] <= 0 or self.geom['box'][i]['Origin'][1] >= self.geom["length"]: 
                    raise ValueError ("y-Origin of box " + str(i) + " origin is outside the mesh")
                elif self.geom['box'][i]['Length'] >= self.geom["length"] or self.geom['box'][i]['Origin'][1] + self.geom['box'][i]['Length'] <= 0:
                    raise ValueError ("y-length of box " + str(i) + " is outside the mesh")
                lines_list = []
                point_idx +=1
                gmsh.model.geo.add_point(self.geom['box'][i]['Origin'][0] + self.geom['box'][i]['Width'], self.geom['box'][i]['Origin'][1], 0, self.geom['box'][i]['Elem_size'], point_idx)
                point_idx +=1
                gmsh.model.geo.add_point(self.geom['box'][i]['Origin'][0], self.geom['box'][i]['Origin'][1], 0, self.geom['box'][i]['Elem_size'], point_idx)
                point_idx +=1
                gmsh.model.geo.add_point(self.geom['box'][i]['Origin'][0], self.geom['box'][i]['Origin'][1] + self.geom['box'][i]['Length'], 0, self.geom['box'][i]['Elem_size'], point_idx)
                point_idx +=1
                gmsh.model.geo.add_point(self.geom['box'][i]['Origin'][0] + self.geom['box'][i]['Width'], self.geom['box'][i]['Origin'][1] + self.geom['box'][i]['Length'], 0, self.geom['box'][i]['Elem_size'], point_idx)
                line_idx += 1
                gmsh.model.geo.add_line(line_idx, point_idx - 2, point_idx - 3)
                lines_list.append(line_idx)
                line_idx += 1
                gmsh.model.geo.add_line(line_idx, point_idx - 1, point_idx - 2)
                lines_list.append(line_idx)
                line_idx += 1
                gmsh.model.geo.add_line(line_idx, point_idx, point_idx - 1)
                lines_list.append(line_idx)
                line_idx += 1
                gmsh.model.geo.add_line(line_idx, point_idx - 3, point_idx)
                lines_list.append(line_idx)
                gmsh.model.geo.synchronize()
                gmsh.model.mesh.embed(1, lines_list, 2, 1)
        # Write BHEs
        # Currently fixed with n=6 nodes
        alpha = 6.134
        point_list = []
        for i in range(n_BHEs):
            if self.BHEs[str(i)]['bhe_x'] <= -0.5*self.geom['width'] or self.BHEs[str(i)]['bhe_x'] >= 0.5*self.geom['width']:
                    raise ValueError ("x-coordinate of bhe " + str(i) + " is outside the mesh")
            if self.BHEs[str(i)]['bhe_y'] <= 0 or self.BHEs[str(i)]['bhe_y'] >= self.geom['length']:
                raise ValueError ("y-coordinate of bhe " + str(i) + " is outside the mesh")
            delta = alpha * float(self.BHEs[str(i)]['bhe_radius'])
            point_idx += 1
            gmsh.model.geo.addPoint(self.BHEs[str(i)]['bhe_x'], self.BHEs[str(i)]['bhe_y'], 0, delta, point_idx) 
            point_list.append(point_idx)
            point_idx += 1
            gmsh.model.geo.addPoint(self.BHEs[str(i)]['bhe_x'], self.BHEs[str(i)]['bhe_y'] - delta, 0, delta, point_idx)
            point_list.append(point_idx)
            point_idx += 1
            gmsh.model.geo.addPoint(self.BHEs[str(i)]['bhe_x'], self.BHEs[str(i)]['bhe_y'] + delta, 0, delta, point_idx)
            point_list.append(point_idx)
            point_idx += 1
            gmsh.model.geo.addPoint(self.BHEs[str(i)]['bhe_x'] + 0.866*delta, self.BHEs[str(i)]['bhe_y'] + 0.5*delta, 0, delta, point_idx)
            point_list.append(point_idx)
            point_idx += 1
            gmsh.model.geo.addPoint(self.BHEs[str(i)]['bhe_x'] - 0.866*delta, self.BHEs[str(i)]['bhe_y'] + 0.5*delta, 0, delta, point_idx)
            point_list.append(point_idx)
            point_idx += 1
            gmsh.model.geo.addPoint(self.BHEs[str(i)]['bhe_x'] + 0.866*delta, self.BHEs[str(i)]['bhe_y'] - 0.5*delta, 0, delta, point_idx)
            point_list.append(point_idx)
            point_idx += 1
            gmsh.model.geo.addPoint(self.BHEs[str(i)]['bhe_x'] - 0.866*delta, self.BHEs[str(i)]['bhe_y'] - 0.5*delta, 0, delta, point_idx)
            point_list.append(point_idx)
        if n_add_points > 0:
            for j in range(n_add_points):
                point_idx += 1
                if self.add_points[str(j)]['x'] <= -0.5*self.geom['width'] or self.add_points[str(j)]['x'] >= 0.5*self.geom['width']:
                    raise ValueError ("x-coordinate of add-point " + str(j) + " is outside the mesh")
                if self.add_points[str(j)]['y'] <= 0 or self.add_points[str(j)]['y'] >= self.geom['length']:
                    raise ValueError ("y-coordinate of add-point " + str(j) + " is outside the mesh")
                gmsh.model.geo.addPoint(self.add_points[str(j)]['x'], self.add_points[str(j)]['y'], 0, self.add_points[str(j)]['delta'], point_idx)
                point_list.append(point_idx)
        gmsh.model.geo.synchronize()
        gmsh.model.mesh.embed(0, point_list, 2, 1)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.)
        gmsh.model.mesh.generate(2)
        gmsh_nodes= gmsh.model.mesh.getNodes()
        gmsh_elements = gmsh.model.mesh.getElements(2)
        self.prism_array = np.zeros((len(gmsh_elements[1][0][:]),7), dtype= int)
        self.node_array = np.zeros((len(gmsh_nodes[0][:]),3))
        for i in range(len(gmsh_nodes[0][:])):
            self.node_array[i][0] = gmsh_nodes[1][i * 3]
            self.node_array[i][1]= gmsh_nodes[1][i * 3 + 1]
            self.node_array[i][2]= gmsh_nodes[1][i * 3 + 2]
        for i in range(len(gmsh_elements[1][0][:])):
            # Materialnumber
            self.prism_array[i][0] = 0 
            # Nodeid's
            self.prism_array[i][1] = gmsh_elements[2][0][i * 3 + 0] - 1
            self.prism_array[i][2] = gmsh_elements[2][0][i * 3 + 1] - 1
            self.prism_array[i][3] = gmsh_elements[2][0][i * 3 + 2] - 1
            self.prism_array[i][4] = 0
            self.prism_array[i][5] = 0
            self.prism_array[i][6] = 0
        gmsh.finalize()
        print(' - 2D mesh created successful')


    def extrude_mesh(self):
        n_layertypes = len(self.layers)
        n_nodes_in_plane = len(self.node_array)
        self.n_elems_in_plane  = len(self.prism_array)
        cnt = 1
        n_layers = 0
        for i in range(n_layertypes):
            n_layers += int(self.layers[repr(i)]['n_elems'])
        self.prism_array = np.append(self.prism_array, np.zeros((self.n_elems_in_plane * (n_layers-1), 7), dtype = int), axis = 0)
        self.node_array = np.append(self.node_array, np.zeros((n_nodes_in_plane * (n_layers),3)), axis = 0)
        self.neighbors = np.zeros((self.n_elems_in_plane * (n_layers),5))
        # Loop over layers
        for i in range(n_layertypes):
            n_elems = int(self.layers[repr(i)]['n_elems'])
            mat_group = self.layers[repr(i)]['mat_group']
            z_shift = self.layers[repr(i)]['elem_thickness']
            if mat_group > self.cnt_mat_groups:
                self.cnt_mat_groups = mat_group
            # Loop over elements per layer
            for j in range(n_elems):
                # Create nodes
                for k in range(n_nodes_in_plane):
                    idx_new = cnt * n_nodes_in_plane + k
                    idx_old = (cnt - 1) * n_nodes_in_plane + k
                    self.node_array[idx_new] = self.node_array[idx_old]
                    self.node_array[idx_new][2] -= z_shift
                # Create elements
                for k in range(self.n_elems_in_plane):
                    if cnt == 1:
                        idx = (cnt - 1) * self.n_elems_in_plane + k
                        self.prism_array[idx][4] = self.prism_array[idx][1] + n_nodes_in_plane
                        self.prism_array[idx][5] = self.prism_array[idx][2] + n_nodes_in_plane
                        self.prism_array[idx][6] = self.prism_array[idx][3] + n_nodes_in_plane
                        self.prism_array[idx][0] = mat_group
                        self.neighbors[idx] = np.array([-1, -1, -1, -1, idx + self.n_elems_in_plane])
                    else:
                        idx_old = (cnt - 2) * self.n_elems_in_plane + k
                        idx = idx_old + self.n_elems_in_plane
                        self.prism_array[idx][0] = mat_group
                        self.prism_array[idx][1] = self.prism_array[idx_old][4]
                        self.prism_array[idx][2] = self.prism_array[idx_old][5]
                        self.prism_array[idx][3] = self.prism_array[idx_old][6]
                        self.prism_array[idx][4] = self.prism_array[idx][1] + n_nodes_in_plane
                        self.prism_array[idx][5] = self.prism_array[idx][2] + n_nodes_in_plane
                        self.prism_array[idx][6] = self.prism_array[idx][3] + n_nodes_in_plane
                        if not idx >= (n_layers - 1) * self.n_elems_in_plane:
                            self.neighbors[idx] = np.array([idx_old, -1, -1, -1, (idx + self.n_elems_in_plane)])
                        else:
                            self.neighbors[idx] = np.array([idx_old, -1, -1, -1, -1])
                    self.faces[repr(idx)] = {}
                    self.faces[repr(idx)]['face_1'] = [self.prism_array[idx][1], self.prism_array[idx][3], self.prism_array[idx][2]]
                    self.faces[repr(idx)]['face_2'] = [self.prism_array[idx][1], self.prism_array[idx][2], self.prism_array[idx][5], self.prism_array[idx][4]]
                    self.faces[repr(idx)]['face_3'] = [self.prism_array[idx][2], self.prism_array[idx][3], self.prism_array[idx][6], self.prism_array[idx][5]]
                    self.faces[repr(idx)]['face_4'] = [self.prism_array[idx][3], self.prism_array[idx][1], self.prism_array[idx][5], self.prism_array[idx][4]]
                    self.faces[repr(idx)]['face_5'] = [self.prism_array[idx][4], self.prism_array[idx][5], self.prism_array[idx][6]]
                cnt += 1
        print(" - Extrusion of 2D mesh successful")


    def compute_BHE_elements(self):
        n_BHEs = len(self.BHEs)
        n_nodes = len(self.node_array)
        elem_sum = 0
        self.bhe_array=[]
        # Loop over BHEs
        for i in range(n_BHEs):
            # Increase material group per BHE
            self.cnt_mat_groups += 1
            bhe_nodes = []
            # Loop over nodes
            for j in range(n_nodes):
                # Check if node is on BHE and copy to BHE node list
                if self.node_array[j][0] == self.BHEs[repr(i)]['bhe_x'] and self.node_array[j][1] == self.BHEs[repr(i)]['bhe_y'] and self.node_array[j][2] <= self.BHEs[repr(i)]['bhe_top'] and self.node_array[j][2] >= self.BHEs[repr(i)]['bhe_bottom']:
                    bhe_nodes.append(j)
            if self.bhe_array== []:
                self.bhe_array = np.zeros(((len(bhe_nodes)-1),3), dtype = int)
            else:
                self.bhe_array = np.append(self.bhe_array, np.zeros(((len(bhe_nodes)-1),3), dtype = int), axis = 0)
            n_BHE_elems = len(bhe_nodes) -1
            # Loop over and create BHE elements
            for j in range(n_BHE_elems): 
                # Assign element data
                self.bhe_array[elem_sum + j][0] = int(self.cnt_mat_groups)
                self.bhe_array[elem_sum + j][1] = bhe_nodes[j]
                self.bhe_array[elem_sum + j][2] = bhe_nodes[j + 1]
            elem_sum += n_BHE_elems
        print(" - BHE meshing successful")


    def node_reordering(self):
        n_corrected_elements = 0
        nElements = len(self.prism_array)
        for i in range(nElements):
            element = self.prism_array[i]
            nBaseNodes = 6
            nFaces = 5
            nFaceNodes = [3, 4, 4, 4, 3]
            face_nodes = [[1, 3, 2, 99], [1, 2, 5, 4], [2, 3, 6, 5], [3, 1, 4, 6], [4, 5, 6, 99]]
            center=[0, 0, 0]
            for j in range(nBaseNodes):
                center[0] += self.node_array[element[j+1]][0]
                center[1] += self.node_array[element[j+1]][1]
                center[2] += self.node_array[element[j+1]][2]
            cc = [center[0]/nBaseNodes, center[1]/nBaseNodes, center[2]/nBaseNodes]
            for k in range(nFaces):
                node_ids = []
                for l in range(nFaceNodes[k]):
                    node_ids.append(element[face_nodes[k][l]])
                x = [self.node_array[node_ids[1]][0], self.node_array[node_ids[1]][1], self.node_array[node_ids[1]][2]]
                cx = [x[0] - cc[0], x[1] - cc[1], x[2] - cc[2]]
                a = [self.node_array[node_ids[0]][0], self.node_array[node_ids[0]][1], self.node_array[node_ids[0]][2]]
                b = [self.node_array[node_ids[1]][0], self.node_array[node_ids[1]][1], self.node_array[node_ids[1]][2]]
                c = [self.node_array[node_ids[2]][0], self.node_array[node_ids[2]][1], self.node_array[node_ids[2]][2]]
                u = [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
                v = [c[0] - b[0], c[1] - b[1], c[2] - b[2]]
                cross_uv = [u[1]*v[2] - u[2]*v[1], u[2]*v[0] - u[0]*v[2], u[0]*v[1] - u[1]*v[0]]
                s = cross_uv[0]*cx[0] + cross_uv[1]*cx[1] + cross_uv[2]*cx[2]
                # Reorder nodes if facenormal points towards cc
                if s >= 0: 
                    for m in range(1, 4):
                        node_a = self.prism_array[i][m]
                        node_b = self.prism_array[i][m+3]
                        self.prism_array[i][m] = node_b
                        self.prism_array[i][m+3] = node_a
                    n_corrected_elements += 1
                    break
        print(" - Reordered " + repr(n_corrected_elements) + " elements")


    def write_soil_temperature_IC(self, temperature_IC):
        self.use_temp_IC = True
        if temperature_IC['type'] == "profile":
            seasonal = temperature_IC['seasonal']
            neutral = temperature_IC['neutral']
            gradient = temperature_IC['gradient']
            surface_temp = temperature_IC['sf_temp']
            self.node_array = np.append(self.node_array, np.zeros((len(self.node_array),1)), axis = 1)
            if "seasonal_curve" in temperature_IC: 
                spline = interpolate.CubicSpline(temperature_IC['seasonal_curve'][0], temperature_IC['seasonal_curve'][1])
                neutral_temp = spline(seasonal)
                for i in range(len(self.node_array)):
                    if self.node_array[i][2] > seasonal:
                        self.node_array[i][3] = spline(self.node_array[i][2])
                    elif self.node_array[i][2] <= seasonal and self.node_array[i][2] >= neutral:
                        self.node_array[i][3] = neutral_temp
                    else:
                        self.node_array[i][3] = neutral_temp + gradient*(abs(self.node_array[i][2]-neutral))
            else:
                neutral_temp = temperature_IC['reference'][1]-abs(temperature_IC['reference'][0]-neutral)*gradient
                for i in range(len(self.node_array)):
                    if self.node_array[i][2] > seasonal:
                        self.node_array[i][3] = surface_temp - (surface_temp - neutral_temp) / abs(seasonal) * abs(self.node_array[i][2])
                    elif self.node_array[i][2] <= seasonal and self.node_array[i][2] >= neutral:
                        self.node_array[i][3] = neutral_temp
                    else:
                        self.node_array[i][3] = neutral_temp + gradient*(abs(self.node_array[i][2]-neutral))
        elif temperature_IC['type'] == "fixed":
            self.node_array = np.append(self.node_array, np.full((len(self.node_array),1), temperature_IC['fix_temp']), axis = 1)
        else: 
            print("No temperature IC was applied, because no type of temperature IC was specified.")


    def write_mesh_to_VTK (self):
        writer = vtkXMLUnstructuredGridWriter()
        mesh_data = vtkUnstructuredGrid()
        # Write Nodes and initial soil temperature
        points = vtkPoints()
        if self.use_temp_IC:
            soil_Temperature = np.zeros(len(self.node_array))
            for i in range(len(self.node_array)):
                points.InsertNextPoint([self.node_array[i][0],self.node_array[i][1],self.node_array[i][2]])
                soil_Temperature[i] = self.node_array[i][3]      
            mesh_data.SetPoints(points)
            vtk_temperature = numpy_support.numpy_to_vtk(np.array(soil_Temperature), deep=True, array_type=vtkDoubleArray().GetDataType())
            vtk_temperature.SetName('temperature_soil')
            mesh_data.GetPointData().AddArray(vtk_temperature)
        else:
            for i in range(len(self.node_array)):
                points.InsertNextPoint([self.node_array[i][0],self.node_array[i][1],self.node_array[i][2]])
            mesh_data.SetPoints(points)
        # Write Mesh Elements
        prism_elements = np.zeros((len(self.prism_array),7), dtype=np.int64)
        bhe_elements = self.bhe_array.copy()
        material_ID = []
        element_type = []
        for i in range(len(self.prism_array)):
            prism_elements[i][0] = 6 ## Defines the number of nodes of the Element 
            prism_elements[i][1] = self.prism_array[i][4] ## Nodes were given in the wrong order, because the internal order of the vtu is different
            prism_elements[i][2] = self.prism_array[i][5]
            prism_elements[i][3] = self.prism_array[i][6]
            prism_elements[i][4] = self.prism_array[i][1]
            prism_elements[i][5] = self.prism_array[i][2]
            prism_elements[i][6] = self.prism_array[i][3]
            element_type.append(VTK_WEDGE)
            material_ID.append(self.prism_array[i][0])
        # Write BHE Elements
        for i in range(len(self.bhe_array)):
            bhe_elements[i][0] = 2 ## Defines the number of nodes of the Element 
            element_type.append(VTK_LINE)
            material_ID.append(self.bhe_array[i][0])
        elements = np.append(prism_elements.ravel(),bhe_elements.ravel())
        cells = vtkCellArray()
        cells.SetCells(elements.shape[0],
                numpy_support.numpy_to_vtk(elements, deep=True, array_type=vtkIdTypeArray().GetDataType()))
        mesh_data.SetCells(numpy_support.numpy_to_vtk(element_type, deep=True, array_type=vtkUnsignedCharArray().GetDataType()), cells)
        vtk_material = numpy_support.numpy_to_vtk(np.array(material_ID), deep=True, array_type=vtkIntArray().GetDataType())
        vtk_material.SetName('MaterialIDs')
        mesh_data.GetCellData().AddArray(vtk_material)
        writer.SetFileName(self.prefix + ".vtu")
        writer.SetInputData(mesh_data)
        writer.Write()
        

    def extract_surfaces(self, sf_types=[], IC = []):
        angle = 0 ## tolerance for faceangle
        path = []
        normals = {}   
        normals['Top'] =  {'normal' : [0, 0, 1],
                                'suffix' : "topsf"}
        normals['top'] =  normals['Top']  
        normals['Bottom'] = {'normal' : [0, 0, -1],
                                'suffix' : "bottomsf"}
        normals['bottom'] =  normals['Bottom']  
        normals['Back'] = {'normal' : [0, -1, 0],
                                'suffix' : "backsf"}
        normals['back'] =  normals['Back']  
        normals['Front'] = {'normal' : [0, 1, 0],
                                'suffix' : "frontsf"}
        normals['front'] =  normals['Front']   
        normals['Left'] = {'normal' : [1, 0, 0],
                                'suffix' : "leftsf"}
        normals['left'] =  normals['Left']  
        normals['Right'] = {'normal' : [-1, 0, 0],
                                'suffix' : "rightsf"}
        normals['right'] =  normals['Right']
        
        normals['Back_inflow'] = normals['Back'].copy()
        normals['Back_inflow']['suffix'] = "back_inflowsf"
        normals['back_inflow'] =  normals['Back_inflow']
        normals['Front_inflow'] = normals['Front'].copy()
        normals['Front_inflow']['suffix'] = "front_inflowsf"
        normals['front_inflow'] =  normals['Front_inflow']
        normals['Left_inflow'] = normals['Left'].copy()
        normals['Left_inflow']['suffix'] = "left_inflowsf"
        normals['left_inflow'] =  normals['Left_inflow']
        normals['Right_inflow'] = normals['Right'].copy()
        normals['Right_inflow']['suffix'] = "right_inflowsf"
        normals['right_inflow'] =  normals['Right_inflow']    
            
        for element in range(len(self.prism_array)):
            if element < self.n_elems_in_plane:
                node_1 = np.where(self.prism_array[:,1:]==self.prism_array[element][1])
                node_2 = np.where(self.prism_array[:,1:]==self.prism_array[element][2])
                node_3 = np.where(self.prism_array[:,1:]==self.prism_array[element][3])
                node_4 = np.where(self.prism_array[:,1:]==self.prism_array[element][4])
                node_5 = np.where(self.prism_array[:,1:]==self.prism_array[element][5])
                node_6 = np.where(self.prism_array[:,1:]==self.prism_array[element][6])
                for neighbor in node_1[0]:
                    if neighbor in node_2[0] and neighbor in node_5[0] and neighbor in node_4[0] and neighbor != element:
                        self.neighbors[element][1] = neighbor
                for neighbor in node_2[0]:
                    if neighbor in node_3[0] and neighbor in node_6[0] and neighbor in node_5[0] and neighbor != element:
                        self.neighbors[element][2] = neighbor
                for neighbor in node_3[0]:
                    if neighbor in node_1[0] and neighbor in node_4[0] and neighbor in node_6[0] and neighbor != element:
                        self.neighbors[element][3] = neighbor
            else:
                if not self.neighbors[element-self.n_elems_in_plane][1] == -1:
                    self.neighbors[element][1] = self.neighbors[element-self.n_elems_in_plane][1] + self.n_elems_in_plane
                if not self.neighbors[element-self.n_elems_in_plane][2] == -1:
                    self.neighbors[element][2] = self.neighbors[element-self.n_elems_in_plane][2] + self.n_elems_in_plane
                if not self.neighbors[element-self.n_elems_in_plane][3] == -1:
                    self.neighbors[element][3] = self.neighbors[element-self.n_elems_in_plane][3] + self.n_elems_in_plane
        for surface in sf_types:
            if '_inflow' in surface and 't_aqf' in self.geom:
                if self.geom['t_aqf'] == 0:
                    sf_types.remove(surface)
                    continue
            elif '_inflow' in surface and not 't_aqf' in self.geom:
                raise KeyError("The '_inflow' option can only be used in 'simple' mode")
            suffix = normals[surface]['suffix']
            len_normal = np.sqrt(normals[surface]['normal'][0]**2 + normals[surface]['normal'][1]**2 + normals[surface]['normal'][2]**2)
            norm_dir = [normals[surface]['normal'][0]/len_normal, normals[surface]['normal'][1]/len_normal, normals[surface]['normal'][2]/len_normal]
            cos_theta = np.cos(np.pi*angle / 180)    
            sf_nodes_list = []
            sf_elem_list = []
            sfTriangles = []
            sfQuads = [] 
            for j in range(len(self.prism_array)):            
                if -1 in self.neighbors[j]:
                    for i, neighbor in enumerate(self.neighbors[j]):
                        if neighbor == -1:
                            node_ids = self.faces[repr(j)]['face_' + repr(i+1)]
                            a = [self.node_array[node_ids[0]][0], self.node_array[node_ids[0]][1], self.node_array[node_ids[0]][2]]
                            b = [self.node_array[node_ids[1]][0], self.node_array[node_ids[1]][1], self.node_array[node_ids[1]][2]]    
                            c = [self.node_array[node_ids[2]][0], self.node_array[node_ids[2]][1], self.node_array[node_ids[2]][2]]
                            u = [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
                            v = [c[0] - b[0], c[1] - b[1], c[2] - b[2]]
                            cross_uv = [u[1]*v[2] - u[2]*v[1], u[2]*v[0] - u[0]*v[2], u[0]*v[1] - u[1]*v[0]]
                            len_cross_uv = np.sqrt(cross_uv[0]**2 + cross_uv[1]**2 + cross_uv[2]**2)
                            s = [cross_uv[0]/len_cross_uv, cross_uv[1]/len_cross_uv, cross_uv[2]/len_cross_uv]
                            if s[0]*norm_dir[0] + s[1]*norm_dir[1] + s[2]*norm_dir[2] >= cos_theta:
                                if '_inflow' in surface and -self.geom['z_aqf'] < 0:
                                    max_z = -self.geom['z_aqf']
                                    min_z = -(self.geom['z_aqf'] + self.geom['t_aqf'])
                                    if not self.remove_mesh_elements(max_z, min_z, node_ids):
                                        newnodes = len(sf_nodes_list)
                                        newface = []
                                        for node in node_ids:    
                                            newface.append(newnodes)
                                            newnodes += 1
                                        if len(node_ids) == 3:
                                            sfTriangles.append(newface)
                                        elif len(node_ids) == 4:
                                            sfQuads.append(newface)
                                        sf_elem_list.append(i)
                                        sf_nodes_list.extend(node_ids)
                                else:
                                    newnodes = len(sf_nodes_list)
                                    newface = []
                                    for node in node_ids:    
                                        newface.append(newnodes)
                                        newnodes += 1
                                    if len(node_ids) == 3:
                                        sfTriangles.append(newface)
                                    elif len(node_ids) == 4:
                                        sfQuads.append(newface)
                                    sf_elem_list.append(i)
                                    sf_nodes_list.extend(node_ids)
            writer = vtkXMLUnstructuredGridWriter()
            mesh_data = vtkUnstructuredGrid()
            # Write Nodes
            points = vtkPoints()
            if surface in IC: 
                if self.use_temp_IC:
                    soil_Temperature = np.zeros(len(sf_nodes_list))
                    for i, node in enumerate(sf_nodes_list):
                        points.InsertNextPoint([self.node_array[node][0],self.node_array[node][1],self.node_array[node][2]]) 
                        soil_Temperature[i] = self.node_array[node][3]      
                    mesh_data.SetPoints(points)
                    vtk_temperature = numpy_support.numpy_to_vtk(np.array(soil_Temperature), deep=True, array_type=vtkDoubleArray().GetDataType())
                    vtk_temperature.SetName('temperature_soil')
                    mesh_data.GetPointData().AddArray(vtk_temperature)
                else:
                    for i, node in enumerate(sf_nodes_list):
                        points.InsertNextPoint([self.node_array[node][0],self.node_array[node][1],self.node_array[node][2]])      
                    mesh_data.SetPoints(points)
            else:
                for i, node in enumerate(sf_nodes_list):
                    points.InsertNextPoint([self.node_array[node][0],self.node_array[node][1],self.node_array[node][2]])
                mesh_data.SetPoints(points) 
            bulk_node_IDs = numpy_support.numpy_to_vtk(np.array(sf_nodes_list), deep=True, array_type=vtkTypeUInt64Array().GetDataType())
            bulk_node_IDs.SetName('bulk_node_ids')
            mesh_data.GetPointData().AddArray(bulk_node_IDs)
            # Write Mesh Elements
            tri_elements = np.zeros((len(sfTriangles),4))
            quad_elements = np.zeros((len(sfQuads),5))
            element_type = []
            for i in range(len(sfTriangles)):
                tri_elements[i][0] = 3
                tri_elements[i][1] = sfTriangles[i][0]
                tri_elements[i][2] = sfTriangles[i][1]
                tri_elements[i][3] = sfTriangles[i][2]
                element_type.append(VTK_TRIANGLE)
            for i in range(len(sfQuads)):
                quad_elements[i][0] = 4
                quad_elements[i][1] = sfQuads[i][0]
                quad_elements[i][2] = sfQuads[i][1]
                quad_elements[i][3] = sfQuads[i][2]
                quad_elements[i][4] = sfQuads[i][3]
                element_type.append(VTK_QUAD)
            elements = np.append(quad_elements.ravel(),tri_elements.ravel())
            cells = vtkCellArray()
            cells.SetCells(elements.shape[0],
                    numpy_support.numpy_to_vtk(elements, deep=True, array_type=vtkIdTypeArray().GetDataType()))
            mesh_data.SetCells(numpy_support.numpy_to_vtk(element_type, deep=True, array_type=vtkUnsignedCharArray().GetDataType()), cells)
            bulk_elem_IDs = numpy_support.numpy_to_vtk(np.array(sf_elem_list), deep=True, array_type=vtkTypeUInt64Array().GetDataType())
            bulk_elem_IDs.SetName('bulk_element_ids')
            mesh_data.GetCellData().AddArray(bulk_elem_IDs)
            writer.SetFileName(self.prefix + "_" + suffix + ".vtu")
            writer.SetInputData(mesh_data)
            writer.Write()
            print(" - " + surface + "surface extracted successfully")
            path.append("./" + self.prefix.replace("\\", "/") + "_" + suffix + ".vtu")
        return path


    def remove_mesh_elements(self, zmax, zmin, node_ids):
        for node in node_ids:
            if self.node_array[node][2] > zmax or self.node_array[node][2] < zmin:
                return True
        return False