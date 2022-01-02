import geometric_networks as gn

from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist



path =  ""
file = "test_file_simple.las"

radius = 2.5 #[m]
buffer_size = 20 #[%]

print("Create sorted points.")
points, position = gn.create_points(path+file, voxel_homo = 0.5)


print("Make links.")
links, has_parents = gn.make_links(position, radius)

print("Construct networks.")
gn.construct_paths(points, links, has_parents)

print("Combine networks.")
labels = gn.combine_networks(points, position, buffer_size)

print("Save LAS file.")
gn.save_las("segmented_pointcloud.las", labels, position, labels)









