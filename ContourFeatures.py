import numpy as np
from scipy.spatial import ConvexHull
import math
import cv2
# from ellipsoid_fit import minVolEllipsoid
# from ellipsoid_fit import data_regularize
import plotEllipsoid

def convert3dToSpherical(v1):
        rho = np.sqrt(np.sum(v1 ** 2))
        if v1[0] == 0:
                if v1[1] > 0:
                        theta = 90
                else:
                        theta = -90
        else:
                theta = np.arctan(v1[1] / v1[0]) / np.pi * 180
        phi = np.arccos(v1[2] / rho) / np.pi * 180

        return rho, phi, theta

def addPoints(pts):
        output = {}
        for point in pts:
                for i in range(0, 2):
                        for j in range(0, 2):
                                for k in range(0, 2):
                                        i = i + np.random.rand() * .001
                                        j = j + np.random.rand() * .001
                                        k = k + np.random.rand() * .001
                                        p = point + np.array([i, j, k])
                                        output[str(p)] = p
        # print output
        return np.array(output.values())

class ContourFeatures:
        def __init__(self):
                self.features = {}
                
        def extract2DContoursFrom3D(self, axis=None):
                points = self.hull.points
                if axis == 0:
                        axes = "yz"
                        cnt = np.array([points[:, 1:3]], dtype=np.int32)
                elif axis == 1:
                        axes = "xz"
                        cnt = np.array([points[:, [0, 2]]], dtype=np.int32)
                else:
                        axes = "xy"
                        cnt = np.array([points[:, 0:2]], dtype=np.int32)
                # Contour Perimeter 
                perimeter = cv2.arcLength(cnt,True)
                self.features[axes + "_perimeter"] = perimeter
                # Contour Area 
                area = cv2.contourArea(cnt)
                self.features[axes+"_area"] = area
                # Contour Approximation 
#               epsilon = 0.1*cv2.arcLength(cnt,True)
#               approx = cv2.approxPolyDP(cnt,epsilon,True)
                
                # Convex Hull 
#               hull = cv2.convexHull(cnt)
#               k = cv2.isContourConvex(cnt)

                # Rotated Rectangle
                # rect = cv2.minAreaRect(cnt)
                # box = cv2.boxPoints(rect)
                # box = np.int0(box)
                
#               ellipse = cv2.fitEllipse(cnt)
                # self.features[axes+"_ellipse_center"] = "[%f %f]"%(ellipse[0][0], ellipse[0][1])
                
#               self.features[axes+"_ellipse_radii"] = "[%f %f]"%(ellipse[1][0], ellipse[1][1])
#               self.features[axes+"_ellipse_angle"] = "%f"%ellipse[2]
                
        def printHullCharacteristics(self):
                hull = self.hull
                # print "area: ", hull.area
                # print "volume:", hull.volume
                # print "points:", hull.points
                self.features["convexity"] = 1.0 * self.features["size"] / hull.volume
                self.features["sphericity"] = math.pi ** (1./3) * (6 * hull.volume) ** (2. / 3) / hull.area
                # elongation
                elongation = hull.area**3 / hull.volume**2
                elongation = elongation / (36 * np.pi)
                self.features["elongation"] = elongation
                # self.extract2DContoursFrom3D()
                # self.extract2DContoursFrom3D(0)
                # ellipsoid
                # pts = data_regularize(self.pts, type="cubic")
                # self.ellipsoid = ellipsoid_fit(pts)
                # centers, radii, evecs, v = ellipsoid_fit(pts)
                # self.ellipsoid = minVolEllipsoid(pts.T, 0.01)
                ET = plotEllipsoid.EllipsoidTool()
                self.ellipsoid = ET.getMinVolEllipse(self.pts, .01)
                centers, radii, evecs = self.ellipsoid
                self.features["center_x"] = centers[0]
                self.features["center_y"] = centers[1]
                self.features["center_z"] = centers[2]
                self.features["ellipsoid_r_x"] = radii[0]
                self.features["ellipsoid_r_y"] = radii[1]
                self.features["ellipsoid_r_z"] = radii[2]
                # rho, phi, theta = convert3dToSpherical(evecs[np.argmax(radii)])
                rho, phi, theta = convert3dToSpherical(evecs[0])
                self.features["ellipsoid_phi1"] = phi
                self.features["ellipsoid_theta1"] = theta
                rho, phi, theta = convert3dToSpherical(evecs[1])
                self.features["ellipsoid_phi2"] = phi
                self.features["ellipsoid_theta2"] = theta
                rho, phi, theta = convert3dToSpherical(evecs[2])
                self.features["ellipsoid_phi3"] = phi
                self.features["ellipsoid_theta3"] = theta
                # self.extract2DContoursFrom3D()
                # self.extract2DContoursFrom3D(0)

        def addContour(self, pts):
                #print "started adding contour of size %d...\n"%(len(pts)),
                if len(pts) == 0:
                    return -1
                self.pts = addPoints(pts)
                self.features["size"] = len(pts)
                try:
                        self.hull = ConvexHull(self.pts)
                        self.printHullCharacteristics()
                except Exception as e:
                        print "error has occured for points:", pts
                        print e
                #print("finished!")
        
        def printFeatureKeys(self):
                for key in self.features:
                        print "%s, "%key,
                print

        def printFeatures(self):
                for key in self.features:
                        print "%s, "%self.features[key],
                print

        def writeFeaturesToFile(self, output_file):
                #for key in self.features:
                #      print "%s, "%key,
                #print
                for key in self.features:
                        output_file.write("%s, "%self.features[key],)
                output_file.write("\n")

        def writeKeysToFile(self, output_file):
                for key in self.features:
                        output_file.write("%s, "%key,)
                output_file.write("\n")
        def isEmpty(self):
            if len(self.features) == 0:
                return True
            return False
