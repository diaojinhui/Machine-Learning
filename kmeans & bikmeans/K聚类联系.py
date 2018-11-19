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
    return mat(dataMat)

def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2)))

def randCent(dataSet, k):
    n = shape(dataSet)[1]   #  2
    centroids = mat(zeros((k,n)))    #   2行2列
    for j in range(n):
        minJ = min(dataSet[:,j])
        rangeJ = float(max(dataSet[:,j]) - minJ)
        centroids[:,j] = minJ + rangeJ * random.rand(k,1) # 两行一列的误差
    return centroids  #

def KMeans(dataSet,k):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))
    centroids = randCent(dataSet,k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = inf
            minIndex = -1
            for j in range(k):
                distJI = distEclud(centroids[j,:],dataSet[j,:])
                if distJI<minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i,0] != minIndex:
                clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist ** 2
        for cent in range(k):
            ptsInclust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]
            centroids[cent,:] = mean(ptsInclust,axis=0)
        return centroids,clusterAssment

def testKmeans(k):
    dataMat = mat(loadDataSet('testSet.txt'))
    myCentroids, clustAssing = KMeans(dataMat, k)
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














