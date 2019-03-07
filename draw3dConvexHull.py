from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull
import ContourFeatures
import sys, getopt
import plotEllipsoid

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::
    >>> angle_between((1, 0, 0), (0, 1, 0))
    1.5707963267948966
    >>> angle_between((1, 0, 0), (1, 0, 0))
    0.0
    >>> angle_between((1, 0, 0), (-1, 0, 0))
    3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def angle_between(middle, p1, p2):
    v1 = p1 - middle
    v2 = p2 - middle
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

    >>> angle_between((1, 0, 0), (0, 1, 0))
    1.5707963267948966
    >>> angle_between((1, 0, 0), (1, 0, 0))
    0.0
    >>> angle_between((1, 0, 0), (-1, 0, 0))
    3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
def plotPoints(pts, ax):
        # pts = ContourFeatures.addPoints(pts)(pts)
        pts = ContourFeatures.addPoints(pts)
        pts = np.array(pts)
        ax.scatter(pts.T[0], pts.T[1], pts.T[2], ".", s=.01, color='red')

def plotPointsAndLines(pts, ax, v=None):
        if (len(pts) == 0):
                return
        pts = ContourFeatures.addPoints(pts)
        pts = np.array(pts)
        # ax.scatter(pts.T[0], pts.T[1], pts.T[2], "o", s=5)
        ax.plot(pts.T[0], pts.T[1], pts.T[2], "ko")
        hull = ConvexHull(pts)
        for s in hull.simplices:
                        s = np.append(s, s[0])  # Here we cycle back to the first coordinate
                        ax.plot(pts[s, 0], pts[s, 1], pts[s, 2], "r-")
        if v == None:
                return hull.volume
        return v + hull.volume

def drawEllipsoidVectors(ellipsoid, ax):
        center, radii, evecs = ellipsoid        
        v1 = evecs[0] * radii[0]
        v2 = evecs[1] * radii[1]
        v3 = evecs[2] * radii[2]
        # print "v1 length = ", np.sqrt(np.sum(v1 * v1))
        # print "v2 length = ", np.sqrt(np.sum(v2 * v2))
        # print "v3 length = ", np.sqrt(np.sum(v3 * v3))
        ax.plot([center[0], center[0] + v1[0]], [center[1], center[1] + v1[1]], [center[2], center[2] + v1[2]])
        ax.plot([center[0], center[0] + v2[0]], [center[1], center[1] + v2[1]], [center[2], center[2] + v2[2]])
        ax.plot([center[0], center[0] + v3[0]], [center[1], center[1] + v3[1]], [center[2], center[2] + v3[2]])

def plot3Dcontour(pts, ax, alpha=1, color='blue'):
        if len(pts) == 0:
                return
        pts = ContourFeatures.addPoints(pts)            
        hull = ConvexHull(pts)
        # printHullCharacteristics(hull)
        for (i1, i2, i3) in hull.simplices:
                p1 = pts[i1]
                p2 = pts[i2]
                p3 = pts[i3]
                x = [p1[0], p2[0] - .01, p3[0] + .01]
                y = [p1[1], p2[1] - .01, p3[1]+.01]
                z = [p1[2], p2[2], p3[2]]
                ax.plot_trisurf(x, y, z, linewidth=0, alpha=alpha, color=color)

def plotCubicPoint(pt, ax):
        X = [pt[0], pt[0], pt[0] + 1, pt[0] + 1]
        Y = [pt[1], pt[1]+1, pt[1], pt[1]+1]
        Z = [pt[2], pt[2], pt[2], pt[2]]
        ax.plot_trisurf(X, Y, Z, linewidth=0)
        
        X = [pt[0], pt[0], pt[0] + 1, pt[0] + 1]
        Y = [pt[1], pt[1]+1, pt[1], pt[1]+1]
        Z = [pt[2]+1, pt[2]+1, pt[2]+1, pt[2]+1]
        ax.plot_trisurf(X, Y, Z, linewidth=0)

        X = [pt[0], pt[0], pt[0], pt[0]]
        Y = [pt[1], pt[1], pt[1]+1, pt[1]+1]
        Z = [pt[2], pt[2]+1, pt[2], pt[2]+1]
        ax.plot_trisurf(X, Y, Z, linewidth=0)

        X = [pt[0], pt[0], pt[0]+1, pt[0]+1]
        Y = [pt[1], pt[1], pt[1], pt[1]]
        Z = [pt[2], pt[2]+1, pt[2], pt[2]+1]
        ax.plot_trisurf(X, Y, Z, linewidth=0)

        X = [pt[0]+1, pt[0]+1, pt[0]+1, pt[0]+1]
        Y = [pt[1], pt[1], pt[1]+1, pt[1]+1]
        Z = [pt[2], pt[2]+1, pt[2], pt[2]+1]
        ax.plot_trisurf(X, Y, Z, linewidth=0)

        X = [pt[0], pt[0], pt[0]+1, pt[0]+1]
        Y = [pt[1]+1, pt[1]+1, pt[1]+1, pt[1]+1]
        Z = [pt[2], pt[2]+1, pt[2], pt[2]+1]
        ax.plot_trisurf(X, Y, Z, linewidth=0)

def draw_cylinder(ax, height, radius, upper_center):
    # Cylinder
    x=np.linspace(upper_center[0] - radius, upper_center[0]+radius, 1000)
    z=np.linspace(upper_center[2] - height, upper_center[2], 1000)
    Xc, Zc=np.meshgrid(x, z)
    Yc = np.sqrt(radius**2-(Xc-upper_center[0])**2)

    # Draw parameters
    rstride = 20
    cstride = 10
    ax.plot_surface(Xc, upper_center[1]+Yc, Zc, alpha=0.2, rstride=rstride, cstride=cstride, linewidth=0, color='blue')
    ax.plot_surface(Xc, upper_center[1]-Yc, Zc, alpha=0.2, rstride=rstride, cstride=cstride, linewidth=0, color='blue')

def addBorder(ax, border_file_name=None):
    print border_file_name
    if border_file_name == "old_1":
        addBorder(ax, "old_2")
    elif border_file_name == "old_2":
        shift = np.array([-40, 0, 0])
        p1 = np.array([[100, 50, 200], [100, 150, 200], [200, 50, 200], [200, 150, 200], [100, 50, 50], [100, 150, 50], [200, 50, 50], [200, 150, 50]])
        plot3Dcontour(p1+shift, ax, alpha=.2)
        p1 = np.array([[100, 50, 200], [100, 150, 200], [200, 50, 200], [200, 150, 200], [50, 0, 250], [50, 200, 250], [250, 0, 250], [250, 200, 250]])
        plot3Dcontour(p1+shift, ax, alpha=.2)
        p1 = np.array([[100, 50, 50], [100, 150, 50], [200, 50, 50], [200, 150, 50], [50, 0, 0], [50, 200, 0], [250, 0, 0], [250, 200, 0]])
        plot3Dcontour(p1+shift, ax, alpha=.2)
    elif border_file_name == "nonhip_2.9":
        p1 = np.array([[193, 255, 0], [747, 189, 0], [242.4, 806, 0], [815, 753, 0], [193, 255, 997], [747, 189, 997], [242.4, 806, 997], [815, 753, 997]])
        plot3Dcontour(p1, ax, alpha=.2)
    elif border_file_name == "nonhip_0.7_2.9":
        addBorder(ax, "nonhip_2.9")
        center = np.array([506, 492, 986])
        height = 986
        radius = 464
        conv = np.array([358, 341, 458])
        new_center = conv + center * .7 / 2.9
        new_height = height * .7 / 2.9
        new_radius = radius * .7 / 2.9
        draw_cylinder(ax, new_height, new_radius, new_center)
    elif border_file_name == "nonhip_0.7":
        #plot3Dcontour(points, ax)
        center = np.array([506, 492, 986])
        height = 986
        radius = 464
        print center, radius
        draw_cylinder(ax, height, radius, center)
        #step = 100
        #for i in np.arange(0, step, len(points)-1):
        #    x = [points[i][0], points[i+step][0], points[i][0], points[i+step][0]]
        #    y = [points[i][1], points[i+step][1], points[i][1], points[i+step][1]]
        #    z = [points[i][2], points[i+step][2], 0, 0]
        #    print "x, y and z = ", x, y, z
        #    ax.plot_trisurf(x, y, z, linewidth=0, alpha=.2, color='blue')

def findElligibleClusters(filename, min_size=0, max_size=1000, minX=0, maxX=10000, minY=0, maxY=10000, minZ=0, maxZ=10000):
        print("Started to find elligible clusters ...")
        pore_file = open(filename, 'r')
        print("Opened file %s")%(filename)
        size = None
        for line in pore_file:
            splitted = [int(x) for x in line.split(',')]
            size = splitted[0]
        if size == None:
            return
        clusters = [ [] for i in range(size+1) ]
        print "Length of clusters: %d"%len(clusters)
        #print clusters
        pore_file = open(filename, 'r')
        for line in pore_file:
            splitted = [int(x) for x in line.split(',')]
            x, y, z = splitted[1:4]
            if z < minZ or z > maxZ or y < minY or y > maxY or x < minX or x > maxX:
                continue
            clusters[splitted[0]].append(splitted)
        eligible_cluster_numbers = []
        for cluster in clusters:
            if len(cluster) < min_size or len(cluster) > max_size:
                continue
            eligible_cluster_numbers.append(cluster[0][0])
        #print "eligible_cluster_numbers:", eligible_cluster_numbers
        #print clusters
        return eligible_cluster_numbers




        
def findElligibleClusters2(filename, min_size=0, max_size=1000, minX=0, maxX=10000, minY=0, maxY=10000, minZ=0, maxZ=10000):
        print("Started to find elligible clusters ...")
        temp_size = 1
        current_cluster_number = 1
        eligible_cluster_numbers = []
        pore_file = open(filename, 'r')
        print("Opened file %s")%(filename)
        for line in pore_file:
                splitted = [int(x) for x in line.split(',')]
                x, y, z = splitted[1:4]
                if z < minZ or z > maxZ or y < minY or y > maxY or x < minX or x > maxX:
                        continue
                cluster_number = splitted[0];
                if cluster_number == current_cluster_number:
                        temp_size = temp_size + 1
                else:
                        if temp_size >= min_size and temp_size <= max_size:
                                eligible_cluster_numbers.append(current_cluster_number)
                                # print("current cluster number: %d"%current_cluster_number)
                                # print("cluster number: %d"%cluster_number)
                                # print("count: %d"%temp_size)
                        current_cluster_number = cluster_number
                        temp_size = 1
        pore_file.close();
        # print eligible_cluster_numbers
        # eligible_cluster_numbers.remove(eligible_cluster_numbers[0])
        # print "Elligible Cluster numbers:", eligible_cluster_numbers
        print eligible_cluster_numbers
        print("Finished finding elligible clusters")
        return eligible_cluster_numbers

def drawContours(filename, eligible_cluster_numbers, ax):
    print("started adding contours from %s to plt ...\n"%(filename))
    if len(eligible_cluster_numbers) == 0:
            print("no clusters found. exiting.")
            return
    pore_file = open(filename, 'r')
    for line in pore_file:
        splitted = [int(x) for x in line.split(',')]
        size = splitted[0];
    clusters = [ [] for i in range(size+1) ]
    eligibles = [ False for i in range(size+1)]
    for i in eligible_cluster_numbers:
        eligibles[i] = True
    pore_file = open(filename, 'r')
    for line in pore_file:
        splitted = [int(x) for x in line.split(',')]
        cluster_number = splitted[0];
        if eligibles[cluster_number]:
            clusters[cluster_number].append(splitted[1:4])
    for i in range(size+1):
        if not eligibles[i]:
            continue
        print "cluster number=", i
        cluster = clusters[i]
        print "cluster size=", len(cluster)
        centerOfMass = np.mean(np.array(cluster).reshape(-1, 3), axis=0)
        print "cluster center of mass:", centerOfMass
        #To make sure the points are withing the confines
        #of a particular set of points
        #p1 = np.array([[193, 255, 0], [747, 189, 0], [242.4, 806, 0], [815, 753, 0]])
        #print p1
        #print "angles:"
        #centerOfMass[2] = 0
        #angles = np.zeros(4)
        #for i in range(0, 4):
        #    angles[i] = angle_between(centerOfMass, p1[i, :], p1[(i+1)%4, :])
        #print angles
        #sumOfAngles = np.sum(angles)
        #print "sum of angles: ", sumOfAngles
        #if sumOfAngles/np.pi < 2.3:
        #    continue
        #print "cluster is ", cluster
        #plot3Dcontour(cluster, ax)
        # uncomment to plot the very points in the pores
        plotPoints(cluster, ax)



def drawContours2(filename, eligible_cluster_numbers, ax):
        print("started adding contours from %s to plt ...\n"%(filename))
        pore_file = open(filename, 'r')
        if len(eligible_cluster_numbers) == 0:
                print("no clusters found. exiting.")
                return
        current_cluster_number = eligible_cluster_numbers[0]
        count = 0
        contours = []
        for line in pore_file:
                splitted = [int(x) for x in line.split(',')]
                cluster_number = splitted[0];
                if cluster_number == current_cluster_number:
                        contours.append([splitted[1], splitted[2], splitted[3]])
                elif cluster_number > current_cluster_number:
                        #plotting contours of the given points
                        plot3Dcontour(contours, ax, alpha=.8, color='red')
                        print "cluster number=", cluster_number
                        #plotting minimum bounding ellipsoid
                        #CF = ContourFeatures.ContourFeatures()
                        #flag = CF.addContour(contours)
                        #if flag != -1:
                        #    ET = plotEllipsoid.EllipsoidTool()
                        #    (center, radii, rotation) = CF.ellipsoid
                        #    ET.plotEllipsoid(center, radii, rotation, ax=ax, plotAxes=True)
                        # uncomment to plot the very points in the pores
                            #plotPoints(contours, ax)
                        count = count + 1
                        if count >= len(eligible_cluster_numbers):
                                break
                        current_cluster_number = eligible_cluster_numbers[count]
                        contours = []
        # ax.set_xlabel('X Label')
        # ax.set_ylabel('Y Label')
        # ax.set_zlabel('Z Label')
        print("Finished adding contours to plt.")
        
def extractFetures(filename, eligible_cluster_numbers, output_file_name, print_option=True):
        print("started to print features in %s to %s"% (filename, output_file_name))
        pore_file = open(filename, 'r')
        f = open(output_file_name, 'w')
        if len(eligible_cluster_numbers) == 0:
                print("no clusters found. exiting.")
                f.close()
                return
        current_cluster_number = eligible_cluster_numbers[0]
        count = 0
        contours = []
        cfSet = []
        for line in pore_file:
                splitted = [int(x) for x in line.split(',')]
                cluster_number = splitted[0];
                if cluster_number == current_cluster_number:
                        contours.append([splitted[1], splitted[2], splitted[3]])
                elif cluster_number > current_cluster_number:
                    print("length of contours: %d"%len(contours))
                    CF = ContourFeatures.ContourFeatures()
                    CF.addContour(contours)
                    if not CF.isEmpty():
                        cfSet.append(CF)
                    count = count + 1
                    if count >= len(eligible_cluster_numbers):
                            break
                    current_cluster_number = eligible_cluster_numbers[count]
                    contours = []
        if len(cfSet) != 0:
            cfSet[0].writeKeysToFile(f)
            for CF in cfSet:
                if print_option:
                        CF.printFeatures()
                CF.writeFeaturesToFile(f)
        f.close()
        return cfSet
        
def main(argv):
        # initiation
        helpMessage = '''Usage:
        draw3dConvexHull --opts <options>
where available options are:
  --file <file name>
  --min_size<> >>>>> minimum size of the pore in pixels
  --max_size<>  >>>>> maximum size of the pore in pixels
  --minX (--minY, --minZ)<> >>>>> minimum x (y, or z) value from which pores are considered
  --maxX (--maxY, --maxZ)<> >>>>> maximum x (y, or z)value beyond which pores are not considered
  --drawBorder >>>>> whether to draw the part itself or not     
'''
        filename = ""
        minX = minY = minZ = maxX = maxY = maxZ = output_file_name = min_size = max_size = drawBorder = ""
        try:
                opts, args = getopt.getopt(argv, "h", ["min_size=", "max_size=", "minX=", "maxX=", "minY=", "maxY=", "minZ=", "maxZ=", "file=", "drawBorder=", "output="])
        except getopt.GetoptError:
                print helpMessage
                sys.exit(2)
        for opt, arg in opts:
                if opt == "-h":
                        print helpMessage
                        sys.exit(2)
                if opt == "--min_size":
                        min_size = int(arg)
                if opt == "--max_size":
                        max_size = int(arg)
                if opt == "--minX":
                        minX = int(arg)
                if opt == "--maxX":
                        maxX = int(arg)
                if opt == "--minY":
                        minY = int(arg)
                if opt == "--maxY":
                        maxY = int(arg)
                if opt == "--minZ":
                        minZ = int(arg)
                if opt == "--maxZ":
                        maxZ = int(arg)
                if opt == "--drawBorder":
                        drawBorder = arg
                if opt == "--file":
                        filename = arg
                if opt == "--output":
                        output_file_name = arg
        if filename == "":
                print helpMessage
                sys.exit(2)
        if output_file_name == "":
                output_file_name = "mardas.csv"
                print("output file name is not determined. defaulting to %s ..."%(output_file_name))

        if drawBorder == "":
                print("drawBorder is not determined. defaulting to  %s..."%(drawBorder));
        if min_size == "":
                min_size = 300
                print("min size is not determined. defaulting to  %d..."%(min_size));
        if max_size == "":
                max_size = 1000
                print("max size is not determined. defaulting to  %d..."%(max_size));
        if minX == "":
                minX = 1
                print("min x is not determined. defaulting to  %d..."%(minX));
        if minY == "":
                minY = 1
                print("min y is not determined. defaulting to  %d..."%(minY));
        if minZ == "":
                minZ =1
                print("min z is not determined. defaulting to  %d..."%(minZ));
        if maxX == "":
                maxX = 10000
                print("max x is not determined. defaulting to  %d..."%(maxX));
        if maxY == "":
                maxY = 10000
                print("max y is not determined. defaulting to  %d..."%(maxY));
        if maxZ == "":
                maxZ = 10000
                print("max z is not determined. defaulting to  %d..."%(maxZ));
                      
        fig = plt.figure(figsize=(10, 30))
        ax = fig.add_subplot(111, projection='3d')
        eligible_cluster_numbers = findElligibleClusters(filename, min_size, max_size, minX, maxX, minY, maxY, minZ, maxZ)
        #there are two implementations of drawContours, what 
        #are the differences?
        drawContours2(filename, eligible_cluster_numbers, ax)
        #drawContours(filename, eligible_cluster_numbers, ax)

        #print("writing features to file %s" % output_file_name)
        #cfSet = extractFetures(filename, eligible_cluster_numbers, output_file_name, print_option=True)
        #print("finished writing")

        #addBorder(ax, '/home/bzr0014/Additive/LSU_NonHIP/0.7um/perimeter.dat')
        addBorder(ax, drawBorder)
        #addBorder(ax)
        plt.savefig('./foo.png')
        print("drawing contours")
        plt.show()
        print("finished drawing contours")


if __name__ == "__main__":
        main(sys.argv[1:])
