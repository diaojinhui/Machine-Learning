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
    print(centroids)
    for j in range(n):
        minJ = min(dataSet[:,j])
        rangeJ = float(max(dataSet[:,j]) - minJ)
        centroids[:,j] = minJ + rangeJ * random.rand(k,1)  # 两行一列的误差
    return centroids   #  产生的K个点的中心点

def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))
    print(type(clusterAssment))
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):    #   多少行，
            minDist = inf
            minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j,:],dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i,0] != minIndex:
                clusterChanged = True
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


def biKmeans(dataSet, k, distMeas=distEclud):  # K= 3
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))
    centroid0 = mean(dataSet, axis=0).tolist()[0]
    centList =[centroid0]   # 第一族
    for j in range(m):     # 对数据进行遍历
        clusterAssment[j,1] = distMeas(mat(centroid0), dataSet[j,:])**2  # 对每个数据计算 跟第一个族的距离误差
    while (len(centList) < k):
        lowestSSE = inf
        for i in range(len(centList)):  #  i =  0  第一族
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A==i)[0],:]
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
            sseSplit = sum(splitClustAss[:,1])  # 重新分配的点的总误差
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1]) # 就一个族的总误差
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i            #   i = 0  记录最优
                bestNewCents = centroidMat     #  2个族的误差
                bestClustAss = splitClustAss.copy()  #  分配点的误差结果
                lowestSSE = sseSplit + sseNotSplit
        bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList)
        bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit
        print ('the bestCentToSplit is: ',bestCentToSplit)
        print ('the len of bestClustAss is: ', len(bestClustAss))
        centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0]
        centList.append(bestNewCents[1,:].tolist()[0])  #  对centlist 加上新的
        clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:]= bestClustAss
        #  上一次的分配结果的结果的赋值操作。
    return mat(centList), clusterAssment

def testBiKmeans(k):
    '''
    :param k:
    :return:
    '''
    dataMat = mat(loadDataSet('testSet.txt'))
    centList, myNewAssments = biKmeans(dataMat, k)
    print ('centList:\n', centList)
    plt.figure(1, facecolor='white') # 创建一个新图形, 背景色设置为白色
    plt.scatter(array(centList[:, 0]), array(centList[:, 1]), marker='+', alpha=1, s=150)
    for cent in range(k):
        ptsInClust = dataMat[nonzero(myNewAssments[:, 0].A == cent)[0]]
        plt.scatter(array(ptsInClust[:, 0]), array(ptsInClust[:, 1]), marker='o', alpha=1)
    plt.show()

#testKmeans(3)
testBiKmeans(2)
