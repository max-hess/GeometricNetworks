import numpy as np
import laspy as lp
import pylas
import scipy
from scipy.spatial import cKDTree
import math
import matplotlib.pyplot as plt
from scipy.spatial import distance
import sys
from matplotlib.cm import inferno, tab20


## Classes 

class point:

	index = None
	position = None
	paths = [] # list of paths
	network = None
	vec = None
	linked_to = None
	treeID = None

	def __init__(self, index, position):
		self.index = index
		self.position = position

	def add_path(self, path):
		self.paths = np.append(self.paths, path) 

class path:

	index = None
	points = []
	network = None

	def __init__(self, index):
		self.index = index

	def add_point(self, this_point):
		self.points = np.append(self.points, this_point)

	def all_points_position(self):
		points_pos = np.c_[[],[],[]]
		for point in self.points:
			points_pos = np.append(points_pos, np.c_[point.position], axis=0)
		return points_pos
			
class network:

	index = None
	paths = []
	points = []
	top = None

	def __init__(self, index):
		self.index = index

	def add_path(self, path):
		self.paths = np.append(self.paths, path)
		path.network = self
		for point in path.points:
			point.network = self
		self.points = np.append(self.points, path.points)

	def size(self):
		points = np.array(())
		for path in self.paths:
			for point in path.points:
				points = np.append(points, point)
		return len(points)

	def all_paths(self):
		all_paths = np.array(())
		for path in self.paths:
			all_paths = np.append(all_paths, path.index)
		return all_paths

	def all_points_position(self):
		points_pos = np.array(())
		for point in self.points:
			points_pos = np.append(points_pos, point.position)
		points_pos = np.reshape(points_pos, (-1, 3))
		return points_pos


## Functions

# creates point objects
def create_points(path, sorted=True, label=None, voxel_homo = 0.25, normalize = True):

	f = lp.file.File(path, mode="r")
	
	if normalize == True:
		x = f.x[f.classification == 4]
		y = f.y[f.classification == 4]
		z = f.z[f.classification == 4] #- f.z.min()
	else:
		x = f.x
		y = f.y
		z = f.z
	ID = f.pt_src_id[f.classification == 4] # just for validation
	coord = np.c_[x, y, z]
	f.close()


	if voxel_homo != False:
		# get the common nn distance
		tr = cKDTree(coord)
		dd, _ = tr.query(coord, k = 2)
		dc = voxel_homo
		if voxel_homo == True:
			dc = np.median(dd[:,1])
		
		# get a flat voxel index ii

		xi = ((x-x.min())/dc).astype('int')
		yi = ((y-y.min())/dc).astype('int')
		zi = ((z-z.min())/dc).astype('int')
		ii = xi + yi * xi.max() + zi * yi.max() * xi.max()

		# select only the first point of each voxel
		_, sl = np.unique(ii, return_index = True)
		coord = coord[sl,:]
		ID = ID[sl]

	points = []
	if sorted == False:
		ID_sorted = ID
		for i in range(len(coord)):
			thisID = ID[i]
			i = point(i, coord[i])
			i.treeID = thisID
			points.append(i)
	if sorted == True:
		coord_sorted = coord[coord[:,2].argsort()]
		ID_sorted = ID[coord[:,2].argsort()]
		for i in range(len(coord_sorted)):
			thisID = ID_sorted[i]
			i = point(i, coord_sorted[i])
			i.treeID = thisID
			points.append(i)

	index = np.arange(len(coord))
	return np.array(points), coord_sorted

# creates links between the points
def make_links(position, r):

	tree = cKDTree(position)

	from scipy.spatial import distance
	nn = tree.query_ball_point(position, r)
	links = np.array(())
	centroids = np.array(())
	has_parent = np.zeros(len(position))

	for i in range(len(position)):

		# all nearest neighbours of point i 
		this_nn = np.array((nn[i]))

		# nearest neighbours above point i 
		upper_nnbs = this_nn[this_nn > i]

		# min 2 nearest neighbour 
		if len(upper_nnbs) > 1: 

			# get centroit c 
			c = np.average(position[upper_nnbs], axis=0)
			centroids = np.append(centroids, c)

			# nearest neighbour above c 
			upper_nnbs_c = upper_nnbs[position[upper_nnbs, 2] > c[2]]

			# special case: upper neighbour on the same height as c
			# take the nearest
			if len(upper_nnbs_c) == 0:
				dist = scipy.spatial.distance.cdist(position[upper_nnbs],  np.reshape(position[i],(-1,3)), metric="euclidean")
				l = upper_nnbs[np.argmin(dist)]

			else:
				dist = scipy.spatial.distance.cdist(position[upper_nnbs_c],  np.reshape(c,(-1,3)), metric="euclidean")
				l = upper_nnbs_c[np.argmin(dist)]
			links = np.append(links, l)

		# only 1 nearest neighbour above c 
		if len(upper_nnbs) == 1:
			links = np.append(links, upper_nnbs) #i und > in line 215
			centroids = np.append(centroids, position[upper_nnbs])
	
		# no 1 nearest neighbour above c 
		if len(upper_nnbs) == 0:
			links = np.append(links, i) 
			centroids = np.append(centroids, position[i])
			has_parent[i] = 1

	return links.astype("int"), has_parent.astype('int')

def construct_paths(points, link_nn, has_parent):

	networks = []
	all_paths = []

	for p in points:
		current_idx = p.index

		if points[current_idx].paths == []:
			end = False

			# init new path
			new_path = path(len(all_paths)) # len paths as index
			all_paths.append(new_path)

			# add first point to the path
			new_path.add_point(points[current_idx])
			points[current_idx].add_path(new_path)

			# append path
			while end == False: 

				# point has a parent
				if has_parent[current_idx] != 1: 
					
					# make link
					points[current_idx].linked_to = points[link_nn[current_idx]]
					
					if points[current_idx].linked_to.paths == []:

						# not in path 
						points[current_idx].linked_to.add_path(new_path)
						new_path.add_point(points[current_idx].linked_to)
						current_idx = link_nn[current_idx]

					else:
						# in path
						points[current_idx].linked_to.network.add_path(new_path)
						points[current_idx].add_path(new_path)
						points[current_idx].linked_to.add_path(new_path)
						end = True

				# point has no parent
				# make network, end path
				else: 
					points[current_idx].linked_to = points[current_idx]
					# init new network
					new_network = network(len(networks)) # len networks as index
					new_network.add_path(new_path) # path and points are assigned to the network
					new_network.top = current_idx
					new_network.points = new_path.points # add points to the network
					networks.append(new_network)
					points[current_idx].network = new_network
					end = True

	return networks

def get_centroids(labels, position):

	list = np.unique(labels)
	d = 2 #dimensions
	centroids = np.array(())
	size = np.array(())

	for l in list:
		p = position[labels == l, :d]
		c = np.average(p, axis=0)
		centroids = np.append(centroids, c)
		size = np.append(size, len(p))
	centroids = np.reshape(centroids, (-1, 2))

	return centroids, size

def get_search_radius(labels, position, buffer_size):

	list = np.unique(labels)
	d = 2 #dim
	centroids = np.array(())
	radius = np.array(())

	for l in list:
		p = position[labels == l, :d]
		c = np.average(p, axis=0)
		dist = scipy.spatial.distance.cdist(p[:,:d],  np.reshape(c,(1, d)), metric="euclidean")
		perc = np.percentile(dist, 90)
		perc = perc + (perc/100) * buffer_size #buffer in %
		radius = np.append(radius, perc)

	return radius

def get_neighbours(unique_labels, distance_matrix, search_radius):
	neighbors = []

	for i in range(len(unique_labels)):
		n = np.argwhere(distance_matrix[i] <= search_radius[i])
		n = n[n != i]
		neighbors.append(n)

	return neighbors

def join(labels, neighbors, distance_matrix, n_label, counter, position):

	mask = np.zeros((len(neighbors), len(neighbors)), dtype=bool)
	for n in range(len(neighbors)):
		if len(neighbors[n]) >= 1:
			for m in neighbors[n]:
				mask[n,m] = True
	
	# find nearest neighbours
	dm = distance_matrix * mask
	
	np.fill_diagonal(dm, float("inf"))
	dm = np.where(dm == 0.0, float("inf"), dm)
	r, c = np.unravel_index(dm.argmin(), dm.shape)

	uni = np.unique(labels)

	# merge
	labels = np.where(labels == uni[r], n_label+counter, labels)
	labels = np.where(labels == uni[c], n_label+counter, labels)

	return labels, dm

def merge(labels, position, n_label, counter, buffer_size):

	# unique labels
	segments = np.unique(labels)

	# centroids
	centroids, size = get_centroids(labels, position)

	# distances
	distance_matrix = distance.cdist(centroids, centroids, metric="euclidean")

	# neighbours
	search_radius = get_search_radius(labels, position, buffer_size=buffer_size)
	
	neighbors = get_neighbours(segments, distance_matrix, search_radius)
	
	# join networks
	labels, dm = join(labels, neighbors, distance_matrix, n_label, counter, position)

	return labels, dm

def lpred(points):
	labels = np.zeros(len(points))
	for p in points:
		labels[p.index] = p.network.index
	labels = labels.astype('int')
	return labels

def combine_networks(points, position, buffer_size):

	labels = lpred(points)
	end = False
	n_label = labels.max()
	counter = 1
	break_var = 10

	while end == False:
		labels, dm = merge(labels, position, n_label, counter, buffer_size)
		counter += 1
		if len(np.unique(dm)) <= 10:
					
			if len(np.unique(dm)) > break_var:
				end = True
			if len(np.unique(dm)) <= 1:
				end = True
			else:
				break_var = len(np.unique(dm))
	return labels

def read(fn):
    fp = lp.file.File(fn)
    x = fp.x
    y = fp.y
    z = fp.z
    xt = fp.x_t
    yt = fp.y_t
    zt = fp.z_t
    p = np.c_[x, y, z, xt, yt, zt]
    p_veg = np.c_[x, y, z, xt, yt, zt]
    return p[fp.return_num == 1], p_veg[fp.raw_classification == 4]

def save_las(fn, v, p, c, cmap = None):
	if cmap is None:
		from matplotlib.cm import tab20 as cmap
		vals = np.linspace(0,1,100)
		np.random.shuffle(vals)
		cmap = plt.cm.colors.ListedColormap(plt.cm.tab20(vals))
	vf = v.astype('float') - v.min()
	vf /= vf.max()
	rgb = (65535*cmap(vf)[:,:3]).astype('uint')
	header = lp.header.Header()
	header.data_format_id = 2
	fp = lp.file.File(fn, mode = 'w', header = header)
	fp.header.scale = [0.01, 0.01, 0.01]
	fp.header.offset = [p[:,0].min(), p[:,1].min(), p[:,2].min()]
	fp.x = p[:, 0]
	fp.y = p[:, 1]
	fp.z = p[:, 2]
	fp.pt_src_id = c.astype('uint16')
	fp.set_red(rgb[:, 0])
	fp.set_green(rgb[:, 1])
	fp.set_blue(rgb[:, 2])
	fp.close()
