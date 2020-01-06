# -*- coding:UTF-8 -*-
"""
created on Thursday,19:33,2020-01-02
@author:Jeaten
e_mail:ljt_IT@163.com
"""
import math
def judge(actual,prediction):###    判断实际值是否与预测值相等
    return actual==prediction
def distance(point1,point2,p):###   计算两点的Lp距离
    '''
    this function is used to calculate the distance of two points
    :param point1: first point 
    :param point2: second point
    :param p: value p, is used to distinguish all kinds of distances
    :return: the distance under 'p'
    '''
    assert point1.__len__()==point2.__len__()
    dis=0
    for i in range(point1.__len__()):
        dis+=math.pow(abs(point1[i]-point2[i]),p)
    return dis**(1/p)
def generate_data():### 生成数据
    '''
    this function is used to generate the data you need
    :return: the feature and label
    '''
    height_weight=[(1,101),(5,89),(108,5),(115,8)]
    label=[1,1,-1,-1]
    return height_weight,label
def knn(actual,feature,label,k):### knn算法
    '''
    the knn(k-nearest neighbor) algorithm
    :param actual: the actual value you want to know which category it belongs to
    :param feature: the feature of all data
    :param label: the label of all data
    :param k: the value of k 
    :return: the category the actual value most likely belongs to
    '''
    assert  feature.__len__()==label.__len__()
    temp=[]
    for i in range(feature.__len__()):
        temp.append(distance(actual,feature[i],p))
    temp=sorted(enumerate(temp),key=lambda x:x[1])
    res=[label[i[0]] for i in temp]
    return max(res,key=res[:k].count)
def __test__():
    ####    测试其在鸢尾花数据集上的效果
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    load_data = load_iris()
    x = load_data.data
    y = load_data.target
    x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.2 )
    correct = 0
    for i in range( x_test.__len__() ):
        prediction = knn( x_test[i], x_train, y_train, k )
        if judge( y_test[i], prediction ):
            correct += 1
    print( correct, x_test.__len__(), correct / x_test.__len__() )
if __name__ == '__main__':
    p=2
    k=3
    feature,label=generate_data()
    dic={1:"爱情片",-1:"动作片"}###   字典结构，用于显示结果
    act=(8,89)
    print("该片为:",dic[knn(act,feature,label,k)])
    # __test__()
