"""
Seth Kinsaul (smk0036@auburn.edu)
10/15/2020
COMP 5600 Aritifical Intelligence
Homework 5
Hierarchical clustering algorithm on cordinates given in .txt files
K meansa algorithm to evaluate clusters given the number of clusters
"""
#imports
import numpy as np
from matplotlib import pyplot as plt
class data:

    def __init__ (self, file_content, *colors, graph_name):
        self.file_content = file_content
        self.colors = colors
        self.graph_name = graph_name
        self.clusters = [] #list of clusters

    def init_clusters(self):
        for x in self.file_content:
            self.clusters.append([[(x[0],x[1])],0])

    def display_plot(self):
        plt.figure()
        plt.scatter(self.file_content[:,0], self.file_content[:,1], color = "red")
        plt.title("Scatter Plot of %s.txt" % self.graph_name)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()

    def display_clusters(self):
        plt.figure()
        for i,x in enumerate(self.clusters):
            pts = x[0]
            x = []
            y = []
            for pt in pts:
                x.append(pt[0])
                y.append(pt[1])
            plt.scatter(x, y, color = '%s' % colors[i])
        plt.title("Hierarchical Clustering of %s.txt" % self.graph_name)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()

    def hier_cluster(self, k):
        self.init_clusters()
        while len(self.clusters) > k:
            best_d = 9999
            best_pt = self.clusters[0][0]
            best_pt2 = self.clusters[1][0]
            best_distance_x = 0
            best_distance_x2 = 1
            for x in range(len(self.clusters)):
                for y in range(len(self.clusters)):
                    if y == x: continue
                    else:
                        pt = self.clusters[x][0]
                        pt2 = self.clusters[y][0]

                        x_pt = []
                        y_pt = []

                        for i in pt:
                            x_pt.append(i[0])
                            y_pt.append(i[1])

                        x_pt2 = []
                        y_pt2 = []
                        for i in pt2:
                            x_pt2.append(i[0])
                            y_pt2.append(i[1])

                        sub_d_best = 9999
                        for i in range(len(pt)):
                            for j in range(len(pt2)):
                                delx = (x_pt[i]-x_pt2[j])**2
                                dely = (y_pt[i]-y_pt2[j])**2
                                sub_d = np.sqrt(delx+dely)
                                if sub_d <= sub_d_best:
                                    sub_d_best = sub_d

                        d = sub_d_best

                        if d <= best_d:
                            best_d = d
                            best_pt = self.clusters[x]
                            best_pt2 = self.clusters[y]
                            best_distance_x = x
                            best_distance_x2 = y

            temp_pt_list = []
            temp_pt_list = self.clusters[best_distance_x][0] + self.clusters[best_distance_x2][0]
            self.clusters.remove(best_pt)
            self.clusters.remove(best_pt2)
            self.clusters.append([temp_pt_list,best_d])


if __name__  == '__main__':
    #load A.txt and B.txt
    a = np.loadtxt('A.txt')
    b = np.loadtxt('B.txt')
    #colors for clusters
    colors = np.array(["red", "black", "green", "yellow"])

    #set_a
    set_a = data(a, colors, graph_name = 'A')
    set_a.hier_cluster(3)
    set_a.display_plot()
    set_a.display_clusters()

    #set_b
    set_b = data(b, colors, graph_name = 'B')
    set_b.hier_cluster(2)
    set_b.display_plot()
    set_b.display_clusters()
