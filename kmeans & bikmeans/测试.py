from numpy import *
import matplotlib as mpl
import matplotlib.pyplot as plt

def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float,curLine))
        dataMat.append(fltLine)
    return dataMat

def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2)))

def randCent(dataSet, k):
    n = shape(dataSet)[1]   #  2
    centroids = mat(zeros((k,n)))    #   2行2列
    for j in range(n):
        minJ = min(dataSet[:,j])
        rangeJ = float(max(dataSet[:,j]) - minJ)
        centroids[:,j] = minJ + rangeJ * random.rand(k,1) # 两行一列的误差
    return centroids   #  产生的K个点的中心点

def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))
    print(type(clusterAssment))
    centroids = createCent(dataSet, k)

    for i in range(m):    #   多少行，
        minDist = inf
        minIndex = -1
        for j in range(k):
            distJI = distMeas(centroids[j,:],dataSet[i,:])
            if distJI < minDist:
                minDist = distJI
                minIndex = j
        clusterAssment[i,:] = minIndex,minDist**2
    for cent in range(k):
        ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]
        centroids[cent,:] = mean(ptsInClust, axis=0)

    return centroids, clusterAssment

def testKmeans(k):
    dataMat = mat(loadDataSet('testSet.txt'))
    myCentroids, clustAssing = kMeans(dataMat, k)
    print ('myCentroids:\n', myCentroids)
    mpl.rcParams['font.sans-serif'] = [u'SimHei'] # 指定显示字体
    mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像中负号'-'显示为方块的问题
    plt.figure(1, facecolor='white') # 创建一个新图形, 背景色设置为白色
    plt.scatter(array(myCentroids[:, 0]), array(myCentroids[:, 1]), marker='+', alpha=1, s=150)
    for cent in range(k):
        ptsInClust = dataMat[nonzero(clustAssing[:, 0].A == cent)[0]]
        plt.scatter(array(ptsInClust[:, 0]), array(ptsInClust[:, 1]), marker='o', alpha=1)
    plt.show()

testKmeans(2)
