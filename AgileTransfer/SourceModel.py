import numpy as np
from sklearn.linear_model import Ridge
from skglm import MCPRegression
from sklearn.linear_model import LinearRegression
import os
import unified_tester
from sklearn.decomposition import PCA
import xgboost as xgb
import joblib


# class single_boom_trainer:
#     def __init__(self,parameter_list):
#         self.model_1 = MCPRegression(alpha=0.00005,gamma=200)
#         self.model_2 = Ridge(alpha=0)
#         self.selection = []
    
#     def train(self,train_feature,train_label):

#         self.model_1.fit(train_feature,train_label)
#         print ('Signals', (np.abs(self.model_1.coef_) > 0).sum(), (np.abs(self.model_1.coef_) > 1e-5).sum())
#         #prune_vector_0 = np.abs(self.model_1.coef_) > 0
#         #prune_vector_0_index = []
#         #for sigs_idex in range(prune_vector_0.shape[0]):
#         #    if prune_vector_0[sigs_idex]:
#         #        prune_vector_0_index.append(sigs_idex)
#         #self.selection = prune_vector_0_index
#         #train_feature_prune = train_feature[:,self.selection]
#         #self.model_2.fit(train_feature_prune,train_label)
#         return
    
#     def predict(self,test_feature):
#         #test_feature_prune = test_feature[:,self.selection]
#         return self.model_1.predict(test_feature)


# tester = unified_tester.BOOM_Tester(single_boom_trainer,[])
# tester.cross_validation(0,'0_basic_train_tune')
def select_feature_init(feature):
    norm_full = feature.sum(axis=0) 
    index = []
    for i in range(norm_full.shape[0]):
        if norm_full[i]>0:
            index.append(i)
    return index


source_config_id = 0
target_config_id = 4

class single_boom_trainer:
    def __init__(self,parameter_list):
        self.num_of_dims = parameter_list[0]
        #self.model_pca = PCA(n_components=parameter_list[0])
        #self.xgboost = xgb.XGBRegressor(n_estimators=25,max_depth=3)
        #self.xgboost = xgb.XGBRegressor(n_estimators=25)
        self.xgboost = MCPRegression(alpha=0.00005,gamma=600)
        #self.xgboost = xgb.XGBRegressor()
        self.selection = []
        self.nz_select = []
        self.select_index = []
        self.n_fold = parameter_list[1]
    
    def train(self,train_feature,train_label):
        self.select_index = np.load('boom{}select_with_{}.npy'.format(source_config_id,target_config_id))
        self.select_index = self.select_index.tolist()
        train_feature = train_feature[:,self.select_index]
        #self.nz_select = select_feature_init(train_feature)
        #train_feature = train_feature[:,self.nz_select]
        # print("Start PCA")
        # if os.path.exists('pca_model/pca_{}_{}_config_{}_common_zero.pkl'.format(self.num_of_dims,self.n_fold,config_id)):
        #     self.model_pca = joblib.load('pca_model/pca_{}_{}_config_{}_common_zero.pkl'.format(self.num_of_dims,self.n_fold,config_id))
        # else:
        #     self.model_pca.fit(train_feature)
        #     os.system("touch pca_model/pca_{}_{}_config_{}_common_zero.pkl".format(self.num_of_dims,self.n_fold,config_id))
        #     joblib.dump(self.model_pca, 'pca_model/pca_{}_{}_config_{}_common_zero.pkl'.format(self.num_of_dims,self.n_fold,config_id))
        # print("PCA completed")
        # train_feature = self.model_pca.transform(train_feature)
        self.xgboost.fit(train_feature,train_label)
        
        joblib.dump(self.xgboost, 'common_model/common_{}_with_{}_{}.pkl'.format(source_config_id,target_config_id,self.n_fold))
        
        return
    
    def predict(self,test_feature):
        test_feature = test_feature[:,self.select_index]
        #test_feature = test_feature[:,self.nz_select]
        #test_feature = self.model_pca.transform(test_feature)
        return self.xgboost.predict(test_feature)


#tester = unified_tester.BOOM_Tester(single_boom_trainer,[1000])
#tester.cross_validation(0,'0_pca_mcp_1000')

# for i in range(2,8):
#     config_id = i
#     tester = unified_tester.BOOM_Tester(single_boom_trainer,[300])
#     tester.cross_validation(config_id,'{}_pca_mcp_300_common'.format(config_id))

pair_list = [[0,1],[1,0],[0,4],[4,0],[1,4],[4,1]]

for i in range(5,len(pair_list)):
    source_config_id = pair_list[i][0]
    target_config_id = pair_list[i][1]
    tester = unified_tester.BOOM_Tester(single_boom_trainer,[300])
    tester.cross_validation(source_config_id,'{}_with_{}_pca_mcp_300_common'.format(source_config_id,target_config_id))