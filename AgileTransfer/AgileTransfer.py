import numpy as np
from sklearn.linear_model import Ridge
from skglm import MCPRegression
from sklearn.linear_model import LinearRegression
import os
import unified_tester
from sklearn.decomposition import PCA
import xgboost as xgb
import joblib
from sklearn.cluster import KMeans

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


source_config_id = None#0
target_config_id = None#4

sample_strategy = None#'inner_al_from_{}_config_{}'.format(20,target_config_id)

class single_boom_trainer:
    def __init__(self,parameter_list):
        self.num_of_dims = parameter_list[0]
        self.num_of_sample = parameter_list[1]
        self.pre_sampling = parameter_list[2]
        self.config_id = parameter_list[3]
        self.model_pca = PCA(n_components=parameter_list[0])
        #self.xgboost = xgb.XGBRegressor(n_estimators=25,max_depth=3)
        #self.xgboost = xgb.XGBRegressor(n_estimators=25)
        self.xgboost = MCPRegression(alpha=0.0001,gamma=600)
        #self.xgboost = xgb.XGBRegressor()
        self.selection = []
        self.nz_select = []
        self.select_index = []
        self.n_fold = parameter_list[4]
        
        
    def get_selected_sample(self,train_benchmark,num_of_sample):
        if num_of_sample == 800:
            if os.path.exists('pca_cluster_300_{}/{}_{}.npy'.format(target_config_id,num_of_sample,self.n_fold)):
                sample_index = np.load('pca_cluster_300_{}/{}_{}.npy'.format(target_config_id,num_of_sample,self.n_fold)).tolist()
            else:
                print("Sample Error")
                return
        elif num_of_sample == 1:
            if os.path.exists('pca_cluster_300_{}/800_{}.npy'.format(target_config_id,self.n_fold)):
                sample_index = np.load('pca_cluster_300_{}/800_{}.npy'.format(target_config_id,self.n_fold)).tolist()
            else:
                print("Sample Error")
                return
            train_feature_tmp = train_benchmark[sample_index]
            mean_vector = np.average(train_feature_tmp, axis=0)
            distance = np.linalg.norm((train_feature_tmp-mean_vector),axis=1)
            #print(distance.shape)
            target_index = np.argmin(distance)
            sample_index = [target_index]
        else:
            if os.path.exists(sample_strategy+'/{}_{}.npy'.format(num_of_sample,self.n_fold)):
                sample_index = np.load(sample_strategy+'/{}_{}.npy'.format(num_of_sample,self.n_fold)).tolist()
            else:
                print("Sample Error")
                return
        
        distance_table = np.zeros((len(sample_index),train_benchmark.shape[0]))
        for i in range(len(sample_index)):
            distance_vector = np.linalg.norm(train_benchmark - train_benchmark[sample_index[i]], axis=1)
            distance_table[i] = distance_vector
        cluster_labels = np.argmin(distance_table,axis=0)
        print("Cluster Completed")
        return sample_index, cluster_labels.tolist()
    
    
    def train(self,train_feature,train_label):
        self.select_index = np.load('boom{}select_with_{}.npy'.format(target_config_id,source_config_id))
        self.select_index = self.select_index.tolist()
        
        train_feature_for_transfer = train_feature[:,self.select_index]
        tmp_model = joblib.load('common_model/common_{}_with_{}_{}.pkl'.format(source_config_id,target_config_id,self.n_fold))
        train_label_predict = tmp_model.predict(train_feature_for_transfer)
        
        self.nz_select = select_feature_init(train_feature)
        train_feature = train_feature[:,self.nz_select]
        
        print("Start PCA")
        if os.path.exists('pca_model/pca_{}_{}_config_{}.pkl'.format(self.num_of_dims,self.n_fold,self.config_id)):
            self.model_pca = joblib.load('pca_model/pca_{}_{}_config_{}.pkl'.format(self.num_of_dims,self.n_fold,self.config_id))
        else:
            self.model_pca.fit(train_feature)
            print("OK")
            os.system("touch pca_model/pca_{}_{}_config_{}.pkl".format(self.num_of_dims,self.n_fold,self.config_id))
            joblib.dump(self.model_pca, 'pca_model/pca_{}_{}_config_{}.pkl'.format(self.num_of_dims,self.n_fold,self.config_id))
        print("PCA completed")
        
        train_feature = self.model_pca.transform(train_feature)
        
        #diff = np.average(train_label) - np.average(train_label_predict)
        #self.xgboost.intercept_ = self.xgboost.intercept_ + diff
        #print(diff)
        
        #self.xgboost.fit(train_feature,train_label_predict)
        if self.num_of_sample == 0:
            scaled_prediction = train_label_predict
            # elif self.num_of_sample == 1:
            #     scaled_prediction = train_label_predict * (np.average(train_label) / np.average(train_label_predict))
        else:
            closest_sample_indices, cluster_labels = self.get_selected_sample(train_feature,self.num_of_sample)

            selected_label = train_label[closest_sample_indices]
            selected_prediction = train_label_predict[closest_sample_indices]
            selected_ratio = selected_label / selected_prediction
        
            scaled_prediction = np.zeros(train_label.shape)
            for i in range(train_feature.shape[0]):
                scaled_prediction[i] = train_label_predict[i] * selected_ratio[cluster_labels[i]]
        
        self.xgboost.fit(train_feature,scaled_prediction)
        
        return
    
    def predict(self,test_feature):
        #test_feature = test_feature[:,self.select_index]
        test_feature = test_feature[:,self.nz_select]
        test_feature = self.model_pca.transform(test_feature)
        return self.xgboost.predict(test_feature)


#tester = unified_tester.BOOM_Tester(single_boom_trainer,[1000])
#tester.cross_validation(0,'0_pca_mcp_1000')

pair_list = [[0,1],[1,0],[0,4],[4,0],[1,4],[4,1]]
#sample_point = [0,50,100,200,400,800]
sample_point = [1]
for i in range(len(pair_list)):
    source_config_id = pair_list[i][0]
    target_config_id = pair_list[i][1]
    sample_strategy = 'inner_al_from_{}_config_{}'.format(20,target_config_id)
    for j in range(len(sample_point)):
        num_of_sample = sample_point[j]
        tester = unified_tester.BOOM_Tester(single_boom_trainer,[300,num_of_sample,20,target_config_id])
        tester.cross_validation(target_config_id,'{}_pca_mcp_300_transfer_al_{}_from_{}_full_signal'.format(target_config_id,num_of_sample,source_config_id))