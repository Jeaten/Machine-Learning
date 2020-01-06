import  numpy as np
def judge(actual,prediction):
    '''
    该函数用于判断实际值和预测值是否一致
    :param actual: 实际值
    :param prediction: 预测值
    :return: 如果实际值和预测值一致，返回为True，否则返回False
    '''
    return actual==prediction
def generate_data():
    '''
    该函数用于生成数据
    :return: 生成好的数据，特征对应于标签
    '''
    feature=[(1,'S'),(1,'M'),(1,'M'),(1,'S'),(1,'S'),
             (2,'S'),(2,'M'),(2,'M'),(2,'L'),(2,'L'),
             (3,'L'),(3,'M'),(3,'M'),(3,'L'),(3,'L')]
    label=[-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1]
    return feature,label
class bayes:
    ### bayes类，用于bayes相关
    def transform(self,feature):
        '''
        对原始输入的特征进行相应的转换，以更好地进行模型训练
        :param feature: 需要转换的特征
        :return: 转换后的特征 
        '''
        fea_transform=[]
        for seq in feature:
            for i in range(seq.__len__()):
                try:
                    fea_transform[i].append(seq[i])
                except:
                    fea_transform.append([])
                    fea_transform[i].append(seq[i])
        return fea_transform
    def train(self,feature,label):
        '''
        训练函数
        :param feature: 特征集
        :param label: 标签集
        :return: 训练好的模型（相应的条件概率和先验概率）
        '''
        feature=np.array(self.transform(feature))
        label=np.array(label)
        classifier=set(label)
        probability_classifier={}
        probability_feature={}
        feature_record={}
        ### 计算先验概率
        for i in classifier:
            probability_classifier[i]=label.tolist().count(i)/label.__len__()
        for fea in range(feature.__len__()):
            feature_record[fea]=set(feature[fea])
        ### 计算条件概率
        for fea in range(feature.__len__()):
            for f in feature_record[fea]:
                for c in classifier:
                    position= np.where(label==c)### 找到分类为 c 的那些位置
                    numerator=feature[fea][position][feature[fea][position]==f].__len__()### 找到类别为 c，同时该特征为 f 的元素个数
                    denominator=feature[fea][label==c].__len__()
                    try:
                        probability_feature[(fea,c)][f]=numerator/denominator
                    except:
                        probability_feature[(fea,c)]={}
                        probability_feature[(fea,c)][f]=numerator/denominator
        return probability_feature,probability_classifier
    def predict(self,actual,model):
        '''
        预测函数
        :param actual: 想要预测的实际值
        :param model: 训练好的模型
        :return: 预测结果
        '''
        actual=np.array(actual)
        probability_feature,probability_classifier=model
        prediction={}
        for classifier in probability_classifier:
            multi=1
            for i in range(actual.__len__()):
                multi*=probability_feature[(i,classifier)][actual[i]]
            prediction[classifier]=probability_classifier[classifier]*multi
        return max(prediction,key=prediction.get)
if __name__ == '__main__':
    feature,label=generate_data()
    bayes=bayes()
    model=bayes.train(feature,label)
    print("预测结果为：",bayes.predict([1,"S"],model))