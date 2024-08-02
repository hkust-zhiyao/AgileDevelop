import numpy as np
from sklearn.linear_model import Ridge
from skglm import MCPRegression
import os
from sklearn.cluster import KMeans
import unified_tester
import my_plot
from sklearn.decomposition import PCA
import joblib


def select_feature_init(feature):
    norm_full = feature.sum(axis=0) 
    index = []
    for i in range(norm_full.shape[0]):
        if norm_full[i]>0:
            index.append(i)
    return index

class ideal_boom_kmeans_trainer:
    def __init__(self,parameter_list):
        
        self.num_of_dims = 300
        self.model_pca = PCA(n_components=self.num_of_dims)
        self.model_1 = MCPRegression(alpha=0.0001,gamma=600)
        
        self.num_of_sample = parameter_list[0]
        self.pre_sampling = parameter_list[1]
        self.num_of_pool = parameter_list[2]
        self.config_id = parameter_list[3]
        self.nz_select = []
        self.fold_id = parameter_list[4]
        
    # def evaluation(self, predict, label, file_name):
    #     r_report = np.corrcoef(label,predict)[1][0]
    #     mape_report = 0
    #     for i in range(predict.shape[0]):
    #         mape_value = abs((predict[i] - label[i]) / label[i])
    #         if mape_value > 1:
    #             mape_value = 1
    #         mape_report = mape_report + mape_value
    #     mape_report = mape_report / predict.shape[0]
    #     my_plot.plot_power_trace_with_R(predict,file_name,label,r_report,mape_report)
    #     return
    
    def get_selected_sample(self,train_benchmark,num_of_sample):
        if os.path.exists('pca_cluster_300_{}/{}_{}.npy'.format(self.config_id,num_of_sample,self.fold_id)):
            return np.load('pca_cluster_300_{}/{}_{}.npy'.format(self.config_id,num_of_sample,self.fold_id)).tolist()
        else:
            print("Error, no clustering information")
            return
        
    def get_inner_selected_sample(self,sampled_train_benchmark,num_of_sample):
        model = KMeans(n_clusters = num_of_sample)
        model.fit(sampled_train_benchmark)
        cluster_labels = model.predict(sampled_train_benchmark)
        cluster_centers = model.cluster_centers_
        closest_sample_indices = np.zeros(num_of_sample).astype(int)
        distance_table = np.array([100000000 for i in range(num_of_sample)])
        for i in range(sampled_train_benchmark.shape[0]):
            label = cluster_labels[i]
            center = cluster_centers[label]
            dist = np.linalg.norm(center - sampled_train_benchmark[i])
            if dist < distance_table[label]:
                distance_table[label] = dist
                closest_sample_indices[label] = i
        print("Inner Cluster Completed")
        return closest_sample_indices.tolist()
    
    def train(self,train_feature,train_label):
        self.nz_select = select_feature_init(train_feature)
        train_feature = train_feature[:,self.nz_select].astype(float)
        print("Start PCA")
        if os.path.exists('pca_model/pca_{}_{}_config_{}.pkl'.format(self.num_of_dims,self.fold_id,self.config_id)):
            self.model_pca = joblib.load('pca_model/pca_{}_{}_config_{}.pkl'.format(self.num_of_dims,self.fold_id,self.config_id))
        else:
            self.model_pca.fit(train_feature)
            os.system("touch pca_model/pca_{}_{}_config_{}.pkl".format(self.num_of_dims,self.fold_id,self.config_id))
            joblib.dump(self.model_pca, 'pca_model/pca_{}_{}_config_{}.pkl'.format(self.num_of_dims,self.fold_id,self.config_id))
        print("PCA completed")
        train_feature = self.model_pca.transform(train_feature)
        
        if self.num_of_sample == 50:
            sample_select_index = []#self.get_selected_sample(train_feature,self.pre_sampling)
            pool_index = self.get_selected_sample(train_feature,self.num_of_pool)
        else:
            if os.path.exists('inner_al_from_{}_config_{}/{}_{}.npy'.format(self.pre_sampling,self.config_id,self.num_of_sample//2,self.fold_id)):
                sample_select_index = np.load('inner_al_from_{}_config_{}/{}_{}.npy'.format(self.pre_sampling,self.config_id,self.num_of_sample//2,self.fold_id)).tolist()
                pool_index = np.load('inner_al_from_{}_config_{}/pool_{}_{}.npy'.format(self.pre_sampling,self.config_id,self.num_of_sample//2,self.fold_id)).tolist()
            else:
                print("Error, no previous data")
                return
        
        active_sample = self.num_of_sample//2
        if self.num_of_sample == 50:
            active_sample = 50 - self.pre_sampling
            closest_sample_indices = self.get_inner_selected_sample(train_feature[pool_index],self.pre_sampling)
            #print(closest_sample_indices)
            for i in range(self.pre_sampling):
                target_index = closest_sample_indices[i]
                sample_select_index.append(pool_index[target_index])
            
            target_index_list = [closest_sample_indices[i] for i in range(self.pre_sampling)]
            #pool_index.pop(target_index_list)
            pool_index = np.delete(pool_index, target_index_list).tolist()
            
            
        for cur_sample in range(active_sample):
            
            #print(cur_sample)
            
            
            
            train_feature_pre_prune = train_feature[sample_select_index]
            train_label_pre_prune = train_label[sample_select_index]
            train_feature_pool = train_feature[pool_index]
            
            num_of_known = len(sample_select_index)
            num_of_unknown = len(pool_index)
            
            distance_table_x = np.zeros((num_of_unknown,num_of_known))
            for unknown_index in range(num_of_unknown):
                tmp_table = train_feature_pre_prune - train_feature_pool[unknown_index]
                dis_vector = np.linalg.norm(tmp_table,axis=1)
                distance_table_x[unknown_index] = dis_vector
            #print("dist x completed")
            
            tmp_model = MCPRegression(alpha=0.00001,gamma=600)
            tmp_model.fit(train_feature_pre_prune,train_label_pre_prune)
            pred_y = tmp_model.predict(train_feature_pool)
            #print("train completed")
            
            distance_table_y = np.zeros((num_of_unknown,num_of_known))
            for unknown_index in range(num_of_unknown):
                tmp_table = np.abs(train_label_pre_prune - pred_y[unknown_index])
                distance_table_y[unknown_index] = tmp_table
            #print("dist y completed")
            
            distance_table = distance_table_x * distance_table_y
            distance_table = distance_table.min(axis=1)
            target_index = np.argmax(distance_table)
            
            sample_select_index.append(pool_index[target_index])
            pool_index.pop(target_index)
            
        self.model_1.fit(train_feature[sample_select_index],train_label[sample_select_index])
        
        np.save('inner_al_from_{}_config_{}/{}_{}'.format(self.pre_sampling,self.config_id,self.num_of_sample,self.fold_id,self.pre_sampling),np.array(sample_select_index))
        np.save('inner_al_from_{}_config_{}/pool_{}_{}'.format(self.pre_sampling,self.config_id,self.num_of_sample,self.fold_id,self.pre_sampling),np.array(pool_index))
        
        return
    
    def predict(self,test_feature):
        test_feature = test_feature[:,self.nz_select]
        test_feature = self.model_pca.transform(test_feature)
        return self.model_1.predict(test_feature)

# pre_sampling = 100
# num_of_sample = 300
# num_of_pool = 1000
# tester = unified_tester.BOOM_Tester(ideal_boom_kmeans_trainer,[num_of_sample, pre_sampling, num_of_pool, 0])
# tester.cross_validation(0,'0_active_learning_{}_from_{}_train'.format(num_of_sample,num_of_pool))

# num_of_sample = 500
# tester = unified_tester.BOOM_Tester(ideal_boom_kmeans_trainer,[num_of_sample, pre_sampling, num_of_pool, 0])
# tester.cross_validation(0,'0_active_learning_{}_from_{}_train'.format(num_of_sample,num_of_pool))

config_id = 1

pre_sampling = 20
num_of_pool = 800
sample_point = [50,100,200,400]
for i in range(len(sample_point)):
    num_of_sample = sample_point[i]
    tester = unified_tester.BOOM_Tester(ideal_boom_kmeans_trainer,[num_of_sample, pre_sampling, num_of_pool, config_id])
    tester.cross_validation(config_id,'{}_pca_al_inner_{}_from_{}_train_start_from_{}'.format(config_id,num_of_sample,num_of_pool,pre_sampling))