#
#    Created on 1 feb. 2023
#
#    @author: Jorge Carrillo-de-Albornoz <jcalbornoz@lsi.uned.es>
#
#    License: Apache 2.0


"""
        This is the official evaluation package provided for the EXIST 2023 shared task, http://nlp.uned.es/exist2023/. 
        This script allows participants to evaluate their system outputs in all tasks proposed in the evaluation campaign. 
        The formats of the system output as well as a details description of the evaluation methodology is provided 
        in the EXIST Guidelines pdf. 
        
        In order to use the evaluation module participants must execute the following command in a prompt:
        python exist2023evaluation.py 
        
        "-p baselines/EXIST2023_training_task1_baseline_1.json" 
        "-g golds/EXIST2023_training_task1_gold_soft.json" 
        "-e golds/EXIST2023_training_task1_gold_hard.json" 
        "-t task1"
        
        Where the parameters are:
        •    -p: this parameter is mandatory, and indicates the path to the system output with the predictions that we
                 want to evaluate. Notice that no information about the evaluation, hard or soft, must be provided. 
                 The script automatically will deal with this. 
        •    -g: this parameter is mandatory, and indicates the path to the gold standard used in the evaluation. 
                Notice that this gold standard can be the hard or soft one if we only provide one gold standard, that 
                is we do not use the optional parameter -e. If the user provides two gold standard the –g one must be 
                the soft gold standard.
        •    -e: this parameter is optional, and indicates the path to the hard gold standard used in the evaluation.
                 Notice that if this parameter is used, it must link to the hard gold standard. 
        •    -t: this parameter is mandatory, and indicates the task addressed. Options are: “task1”, “task2”, “task3”.            
        
        The metrics here implemented are included also in the PyEvALL evaluation Python module, as well as much more,
         that will be released by the end of 2023. The PyEvALL evaluation module will be also accessible by the end of 
         2023 in the website: http://www.evall.uned.es/.

"""


import json
import pandas as pd
import math
import numpy as np
import sys, getopt
from statistics import NormalDist
import pathlib as p


#Global variables
MONO_LABEL_TASK="mono_label"
MULTI_LABEL_TASK="multi_label"
ID="id"
VALUE="value"

def get_parents_dict(nested_dict, value):
    if nested_dict == value:
        return [nested_dict]
    elif isinstance(nested_dict, dict):
        for k, v in nested_dict.items():
            if k == value:
                return [k]
            p = get_parents_dict(v, value)
            if p:
                return [k] + p
    elif isinstance(nested_dict, list):
        lst = nested_dict
        for i in range(len(lst)):
            p = get_parents_dict(lst[i], value)
            if p:
                return p   

          
def is_child(hierarchy, child):
    if isinstance(hierarchy, dict):
        exist = False
        for c in hierarchy:
            if c ==child:
                return True
            else:
                exist= exist or is_child(hierarchy[c], child)
        return exist    
    elif isinstance(hierarchy, list):
        return child in hierarchy
    


class ICM_Soft(object): 
    """
        The ICM soft metric is an extension of the original of ICM to deal with disagreement evaluations
        developed by Enrique Amigó, Jorge Carrillo-de-Albornoz, Laura Plaza, Julio Gonzalo.
        
    """
       
    def __init__(self, pred_df, gold_df, task, hierarchy):
        #input data
        self.pred_df = pred_df
        self.gold_df = gold_df
        self.task = task 
        self.hierarchy= hierarchy 
        
        #parameters icm
        self.alpha_1=2
        self.alpha_2=2
        self.beta=3 
               
        #data structures for probabilities
        self.gold_average= dict()
        self.gold_deviation= dict()
        self.lst_classes= []  
        self.get_list_classes()
        self.calculate_probabilities()

    #################################################################
    #
    #    Methods to calculate probabilities to compute ICM soft metric.
    #
    #################################################################   
    
    def get_list_classes(self):
        if self.hierarchy==None:
            self.gold_df[VALUE].apply(lambda value: self.search_classes(value))
        else:
            #check for classses not included in hierarchy   
            self.gold_df.apply(lambda row: self.check_class_not_in_hierachy(row, self.hierarchy), axis=1)            
            self.get_classes_hierarchy(self.hierarchy)
            
    
    def search_classes(self, value):
        gold_dict = value
        for c in gold_dict:
            if c not in self.lst_classes:
                self.lst_classes.append(c)  
                  
    
    def check_class_not_in_hierachy(self, gold_row, hierarchy):
        gold_set=[]
        if np.isscalar(gold_row[VALUE]):
            gold_set.append(gold_row[VALUE])
        else:
            gold_set=gold_row[VALUE]  
            
        for c in gold_set:
            if not is_child(hierarchy, c):
                self.hierarchy[c]=[]   
                        
    
    def get_classes_hierarchy(self, hierarchy):
        if isinstance(hierarchy, dict):            
            for c in hierarchy: 
                self.lst_classes.append(c)             
                self.get_classes_hierarchy(hierarchy[c])  
        elif isinstance(hierarchy, list):
            for c in hierarchy:      
                self.lst_classes.append(c) 
                  
    
    ####
    #   Method that calculate the propabilities for each class depending on the type of task.
    ####
    def calculate_probabilities(self):  
        gold_df_extended= self.gold_df.copy()
        
        if not self.hierarchy==None:          
            gold_df_extended[VALUE] = gold_df_extended[VALUE].apply(lambda row: self.propagate_max_weigth_ancestors(row.copy(), self.hierarchy, None))
        
        gold_df_extended[self.lst_classes] =gold_df_extended[VALUE].apply(lambda row: self.expand_df(row))
        gold_df_extended = gold_df_extended.drop(VALUE, axis=1)
            
        sum_items = gold_df_extended.sum().tolist()
        size_gold = len(gold_df_extended)       
        for ind, column in enumerate(gold_df_extended.columns):
            if column==ID:
                continue
            
            self.gold_average[column] = sum_items[ind]/size_gold
            dict_deviation= gold_df_extended[column].value_counts().to_dict()
            deviation= 0
            for v in dict_deviation:
                items_dev = dict_deviation[v] 
                val = float(v)
                deviation = deviation + abs(val-self.gold_average[column])* items_dev
                                
            self.gold_deviation[column]= deviation/size_gold                   
   
 
    ####
    #   Method that propagates the weigth of classes between ancestors in hierarchical evaluations
    ####
    def propagate_max_weigth_ancestors(self, gold_dict, hierarchy, clas):
        if isinstance(hierarchy, dict):            
            for c in hierarchy:                
                self.propagate_max_weigth_ancestors(gold_dict, hierarchy[c], c)
                if not clas==None: 
                    if clas not in gold_dict:
                        gold_dict[clas]= gold_dict[c]
                    else:
                        gold_dict[clas]= max(gold_dict[clas], gold_dict[c])
        elif isinstance(hierarchy, list):
            for c in hierarchy: 
                if c not in gold_dict:
                    gold_dict[c]=0.0                           
                if clas not in gold_dict:
                    gold_dict[clas]= gold_dict[c]
                else:
                    gold_dict[clas]= max(gold_dict[clas], gold_dict[c]) 
                
        return gold_dict  
               
            
    ####
    #    Method that expand the dataframe with all classes. Each column represent a class
    ####
    def expand_df(self, value):
        gold_dict = value
        new_columns=[]
        for c in self.lst_classes:
            if c in gold_dict:
                new_columns.append(gold_dict[c])
            else:
                new_columns.append(0.0)

        return pd.Series(new_columns, index=[self.lst_classes])         
        
 
    #################################################################
    #
    #    Methods to process the metric ICM soft
    #
    #################################################################       
    
    def evaluate(self):
        result_icm = self.gold_df.apply(lambda row: self.calculate_icm_row(row), axis=1).tolist()
        gold_size = len(self.gold_df)
        result = sum(result_icm)/gold_size
        return '%.4f'%(result)
        
        
    def calculate_icm_row(self, gold_row):
        pred_row= self.pred_df[self.pred_df[ID] == gold_row[ID]]
        pred_set=[]
        gold_set=[]
        
        if not pred_row.empty:
            pred_dict = pred_row[VALUE].tolist()[0]
            for c in pred_dict:
                pred_set.append((c,pred_dict[c]))
        
        gold_dict = gold_row[VALUE]
        for c in gold_dict:
            gold_set.append((c,gold_dict[c]))           
        
        union_set= self.union_soft(gold_set, pred_set)
        return self.alpha_1*self.information_content(pred_set) + self.alpha_2*self.information_content(gold_set) - self.beta*self.information_content(union_set)
    
    
    def union_soft(self, set_a, set_b):
        union = []
        for a in set_a:
            union.append(a)
            
        for b in set_b:
            exist=False
            for i, u in enumerate(union):
                if b[0]==u[0]:
                    val =max(b[1], u[1])
                    clas = u[0]
                    union[i]= (clas, val)
                    exist = True
                
            if not exist:
                union.append(b)
        
        return union
                
    
    def information_content(self, classes):
        size = len(classes)
        if size==0:
            return 0
        return self.get_prob_class(classes[0]) + self.information_content(classes[1:size]) - \
        self.information_content(self.calculate_set_deepest_common_ancestor(classes[0], classes[1:size]))
                    
    
    def get_prob_class(self, tupla):
        #Empty set
        if tupla==None or not tupla[0]:
            return 0
        
        #Class does not exist in gold we add minimal information
        if not tupla[0] in self.gold_average:
            return -math.log2(1/len(self.gold_df))        
        else:
            prob = 1-NormalDist(mu=self.gold_average[tupla[0]], sigma=self.gold_deviation[tupla[0]]).cdf(tupla[1])            
            return -math.log2(prob)    
        
    
    def calculate_set_deepest_common_ancestor(self, clas, classes):
        deepest_common_ancestors=[]           
        for c in classes:
            parents_a= get_parents_dict(self.hierarchy, clas[0])
            parents_b= get_parents_dict(self.hierarchy, c[0])
            if parents_a==None or parents_b==None:
                continue
            #intersection list parents, only common are save
            common= [ e for e in parents_a if e in parents_b ]
            size= len(common)
            #select only the deepest parent 
            if size!=0:
                tupla=(common[size-1:][0], min(clas[1], c[1]))
                common = [tupla]
                #Union with previous deepest parents
                deepest_common_ancestors= self.union_soft(deepest_common_ancestors, common) 
                
        return deepest_common_ancestors    
    
    

class ICM_Hard(object):  
    """
            ICM is a similarity function that generalizes Pointwise Mutual Information (PMI), and can be used to 
            evaluate system outputs in classification problems by computing their similarity to the ground 
            truth categories.
            
            ICM metric reference:
                Enrique Amigo and Agustín Delgado. 2022. Evaluating Extreme Hierarchical Multi-label Classification.
                In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 
                pages 5809–5819, Dublin, Ireland. Association for Computational Linguistics.  

    """   
    
    def __init__(self, pred_df, gold_df, task, hierarchy):
        #input data
        self.pred_df = pred_df
        self.gold_df = gold_df
        self.task = task 
        self.hierarchy= hierarchy 
        
        #parameters icm
        self.alpha_1=2
        self.alpha_2=2
        self.beta=3  
        
        #data structures for probabilities
        self.gold_freq= dict() 
        self.gold_prob= dict()
        self.generate_prob()  
           
        
    def generate_prob(self):
        #Mono label classification
        if self.task==MONO_LABEL_TASK:
            #Flat classification
            if self.hierarchy==None:
                self.gold_freq= self.gold_df[VALUE].value_counts().to_dict() 
                gold_size = len(self.gold_df)
                for c in self.gold_freq:
                    self.gold_prob[c]= self.gold_freq[c]/gold_size                    
                    
            #Hierarchical classification
            else:
                self.gold_freq= self.gold_df[VALUE].value_counts().to_dict() 
                gold_size = len(self.gold_df)
            
                self.calculate_prob_hierarchy_mono_label(self.hierarchy)            
                for c in self.gold_freq:
                    self.gold_prob[c]= self.gold_freq[c]/gold_size                    
        
        #Multi label classification            
        elif self.task==MULTI_LABEL_TASK:
            if self.hierarchy==None:
                raise NotImplementedError("Please Implement this method")
            else:  
                #check for classses not included in hierarchy   
                self.gold_df.apply(lambda row: self.check_class_not_in_hierachy(row, self.hierarchy), axis=1)           
                gold_size = len(self.gold_df)
                
                self.calculate_prob_hierarchy_multi_labe(self.hierarchy)           
                for c in self.gold_freq:
                    self.gold_prob[c]= self.gold_freq[c]/gold_size                                   
               
            
    #################################################################
    #
    #    Methods to calculate probabilities to compute ICM hard metric.
    #
    #################################################################         
        
    def calculate_prob_hierarchy_mono_label(self, hierarchy):
        if len(hierarchy)==0:
            return 0
        
        level_freq=0
        if isinstance(hierarchy, dict):
            for c in hierarchy:
                freq_c = 0 if c not in self.gold_freq else self.gold_freq[c]
                value = freq_c + self.calculate_prob_hierarchy_mono_label(hierarchy[c])
                self.gold_freq[c]=value
                level_freq+=value
        elif isinstance(hierarchy, list):
            for c in hierarchy:
                freq_c = 0 if c not in self.gold_freq else self.gold_freq[c]                
                self.gold_freq[c]=freq_c
                level_freq+=freq_c
            
        return level_freq
    
    
    def check_class_not_in_hierachy(self, gold_row, hierarchy):
        gold_set=[]
        if np.isscalar(gold_row[VALUE]):
            gold_set.append(gold_row[VALUE])
        else:
            gold_set=gold_row[VALUE]  
            
        for c in gold_set:
            if not is_child(hierarchy, c):
                self.hierarchy[c]=[]                  
   
    
    def calculate_prob_hierarchy_multi_labe(self, hierarchy):
        if len(hierarchy)==0:
            return 0
        
        if isinstance(hierarchy, dict):
            for c in hierarchy:
                freq_c = sum(self.gold_df.apply(lambda row: self.belgons_item_to_class_or_subclass(row, c, hierarchy[c]), axis=1).tolist())
                self.gold_freq[c]=freq_c
                self.calculate_prob_hierarchy_multi_labe(hierarchy[c])
                
        elif isinstance(hierarchy, list):
            for c in hierarchy:
                freq_c = sum(self.gold_df.apply(lambda row: self.belgons_item_to_class_or_subclass(row, c, []), axis=1).tolist())
                self.gold_freq[c]=freq_c         
        
    
    def belgons_item_to_class_or_subclass(self, gold_row, clas, hierarchy):
        gold_set=[]
        if np.isscalar(gold_row[VALUE]):
            gold_set.append(gold_row[VALUE])
        else:
            gold_set=gold_row[VALUE]
            
        if clas in gold_set:
            return 1
        else:
            for c in gold_set:
                if is_child(hierarchy, c):
                    return 1
        return 0        
        
   
    #################################################################
    #
    #    Methods to process the metric ICM hard
    #
    #################################################################    
    
    def evaluate(self):
        result_icm = self.gold_df.apply(lambda row: self.calculate_icm_row(row), axis=1).tolist()
        gold_size = len(self.gold_df)
        result = sum(result_icm)/gold_size
        return '%.4f'%(result)
       
        
    def calculate_icm_row(self, gold_row):
        pred_row= self.pred_df[self.pred_df[ID] == gold_row[ID]]
        pred_set=[]
        gold_set=[]        

        if not pred_row.empty:
            if np.isscalar(pred_row[VALUE].tolist()[0]):
                pred_set.append(pred_row[VALUE].tolist()[0])
            else:
                pred_set=pred_row[VALUE].tolist()[0]
            
        if np.isscalar(gold_row[VALUE]):
            gold_set.append(gold_row[VALUE])
        else:
            gold_set=gold_row[VALUE]
        
        union_set= list(set(pred_set) | set(gold_set))         
        return self.alpha_1*self.information_content(pred_set) + self.alpha_2*self.information_content(gold_set) - self.beta*self.information_content(union_set)
    
    
    def information_content(self, classes):
        size = len(classes)
        if size==0:
            return 0
        return self.get_prob_class(classes[0]) + self.information_content(classes[1:size]) - \
        self.information_content(self.calculate_set_deepest_common_ancestor( classes[0], classes[1:size]))
                    
    
    def get_prob_class(self, clas):
        #Empty set
        if clas==None:
            return 0
                
        #Class does not exist in gold we add minimal information
        if (not clas in self.gold_prob) or (self.gold_prob[clas]==0.0):
            self.gold_prob[clas]=1/len(self.gold_df)
            return -math.log2(self.gold_prob[clas])        
        
        else:
            return -math.log2(self.gold_prob[clas])
        

    def calculate_set_deepest_common_ancestor(self, clas, classes):
        deepest_common_ancestors=[]   
            
        for c in classes:
            parents_a= get_parents_dict(self.hierarchy, clas)
            parents_b= get_parents_dict(self.hierarchy, c)
            if parents_a==None or parents_b==None:
                continue
            #intersection list parents, only common are save
            common= [ e for e in parents_a if e in parents_b ]
            size= len(common)
            #select only the deepest parent
            common = common[size-1:]
            #Union with previous deepest parents
            deepest_common_ancestors= list(set(deepest_common_ancestors) | set(common))

        return deepest_common_ancestors



class FMeasure(): 
    
    def __init__(self,pred_df, gold_df, task):
        #input data
        self.pred_df = pred_df
        self.gold_df = gold_df
        self.task = task
        
        #parameter
        self.alfa_param=0.5  
        
        #structures
        self.conf_matrix=[]
        self.conf_matrix_mult=dict()
        self.index_classes=dict()
        self.lst_classes=[]        
        
        
    def evaluate(self):        
        #Evaluate the metric for each test case
        classes = self.get_gold_classes()
        results=dict()
        res = 0
        for c in classes:
            TP = self.get_true_positive_per_class(c)
            instances_gold_class= self.get_num_instances_gold_per_class(c)
            instances_pred_class= self.get_num_instances_pred_per_class(c)                
            if (not instances_pred_class==None) and (not instances_pred_class==0):
                p= TP/instances_pred_class
                r= TP/instances_gold_class
                f1 = 1/((self.alfa_param/p) + ((1-self.alfa_param)/r));                        
                results[c]='%.4f'%(f1)
                res += f1 
            else:
                results[c]=0.0            
                
        n = res/len(classes)

        results["marco-F"]='%.4f'%(n) 
        return results  
    
        
    def get_gold_classes(self):
        if len(self.lst_classes)==0:
            if self.task==MULTI_LABEL_TASK:
                lst_classes=[]
                self.gold_df.apply(lambda row: self.get_classes_multilabel(row[VALUE], lst_classes), axis=1)
                self.lst_classes= lst_classes
            else:
                self.lst_classes= self.gold_df[VALUE].unique() 
        
        return self.lst_classes 
    
    
    def get_classes_multilabel(self, value, lst_classes):
        for c in value:
            if c not in lst_classes:
                lst_classes.append(c)
    
    
    def get_true_positive_per_class(self, cl):
        if len(self.conf_matrix)==0 and len(self.conf_matrix_mult)==0:
            self.generate_conf_matrix()
            
        if self.task==MONO_LABEL_TASK:
            index = self.get_index_matrix_by_class(cl)
            return self.conf_matrix[index,index]   
        
        elif self.task==MULTI_LABEL_TASK:
            return self.conf_matrix_mult[cl][1][0]

    
    def generate_conf_matrix(self):
        lst_classes=self.get_gold_classes()
        if self.task==MONO_LABEL_TASK:
            for index, cl in enumerate(lst_classes):
                self.index_classes[cl]=index
                                   
            size = len(self.index_classes)
            self.conf_matrix = np.zeros(shape=(size, size))            
 
            
        elif self.task==MULTI_LABEL_TASK:
            for cl in lst_classes:
                self.conf_matrix_mult[cl]=np.zeros(shape=(2, 2))     
        
        self.gold_df.apply(lambda row: self.generate_pair_matrix(row[ID], row[VALUE], self.pred_df), axis=1)
        
    
    def generate_pair_matrix(self, gold_id, gold_value, pred_df):
        if self.task==MONO_LABEL_TASK:
            row = pred_df[pred_df[ID] == gold_id]
            if not row.empty:
                pred_value= row.iloc[0][VALUE]
                if pred_value in self.index_classes:
                    pos_gold = self.get_index_matrix_by_class(gold_value)
                    pos_pred = self.get_index_matrix_by_class(pred_value)
                    self.conf_matrix[pos_gold, pos_pred]+=+1 
        
        #Binaries confusion matrix for each class like:
        #TN    FP
        #TP    FN
        elif self.task==MULTI_LABEL_TASK:
            row = pred_df[pred_df[ID] == gold_id]
            if not row.empty:
                pred_value= row.iloc[0][VALUE]                  
                for cl in self.conf_matrix_mult:                        
                    #True positive
                    if cl in gold_value and cl in pred_value:
                        self.conf_matrix_mult[cl][1][0]+=1
                    #False negative
                    elif cl in gold_value and not cl in pred_value:
                        self.conf_matrix_mult[cl][1][1]+=1
                    #True negative
                    elif not cl in gold_value and not cl in pred_value:
                        self.conf_matrix_mult[cl][0][0]+=1     
                    #False positive
                    elif not cl in gold_value and cl in pred_value:
                        self.conf_matrix_mult[cl][0][1]+=1                                                                   
                             

    def get_index_matrix_by_class(self, cl):                   
        return self.index_classes[cl]


    def get_num_instances_gold_per_class(self, cl):  
        if self.task==MONO_LABEL_TASK:      
            return self.gold_df[self.gold_df[VALUE] == cl].shape[0]
        
        elif self.task==MULTI_LABEL_TASK:
            occurences = self.gold_df.apply(lambda row: self.is_class_in_array(cl, row[VALUE]), axis=1).tolist()
            return sum(occurences)   
    

    def get_num_instances_pred_per_class(self, cl):
        if self.task==MONO_LABEL_TASK: 
            return self.pred_df[self.pred_df[VALUE] == cl].shape[0]
        
        elif self.task==MULTI_LABEL_TASK:
            occurences = self.pred_df.apply(lambda row: self.is_class_in_array(cl, row[VALUE]), axis=1).tolist()
            return sum(occurences)
        
    
    def is_class_in_array(self, cl, list):
        if cl in list:
            return 1
        return 0
    

class EXIST_2023_evaluation (object): 
    HARD_LABEL_TAG= "hard_label"
    SOFT_LABEL_TAG= "soft_label"
    TASK1_TAG="task1"
    TASK2_TAG="task2"
    TASK3_TAG="task3"
    TASK_1_HIERARCHY = None    
    TASK_2_HIERARCHY = {"YES":["DIRECT","REPORTED","JUDGEMENTAL"], "NO":[]}
    TASK_3_HIERARCHY = {"YES":["IDEOLOGICAL-INEQUALITY","STEREOTYPING-DOMINANCE","OBJECTIFICATION", "SEXUAL-VIOLENCE", "MISOGYNY-NON-SEXUAL-VIOLENCE"], "NO":[]}
        
    
    def __init__(self, pred_file, gold_file, gold_file_hard, task):
        self.pred_dict= self.parser_json(pred_file)
        self.gold_dict= self.parser_json(gold_file)
        self.gold_hard_dict=None
        if gold_file_hard!='':
            self.gold_hard_dict= self.parser_json(gold_file_hard)
        self.task = task
        
        
    def parser_json(self, file):
        try:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)  
            return data 
        except ValueError as e:
            raise Exception("Invalid json format")    
    
    
    def evaluate(self):
        if self.task==self.TASK1_TAG:
            #hard vs hard evaluation
            succes, pred_df,gold_df= self.prepare_data_hard_hard()     
            if succes:       
                icm_hard= ICM_Hard(pred_df, gold_df, MONO_LABEL_TASK, self.TASK_1_HIERARCHY)
                result_icm_hard_hard= icm_hard.evaluate()
                print("TASK 1 - Result ICM evaluation hard-hard:\t", result_icm_hard_hard)
                
                fmeasure= FMeasure(pred_df, gold_df, MONO_LABEL_TASK)
                results = fmeasure.evaluate()
                for r in results:
                    print("TASK 1 - Result FMeasure evaluation hard-hard for class:", r, "=\t", results[r])
            else:
                print("Not valid format for hard-hard evaluation")
            
            #hard vs soft evaluation
            succes, pred_df,gold_df= self.prepare_data_hard_soft()
            if succes: 
                icm_soft = ICM_Soft(pred_df, gold_df, MONO_LABEL_TASK, self.TASK_1_HIERARCHY)
                result_icm_hard_soft=icm_soft.evaluate()
                print("TASK 1 - Result ICM evaluation hard-soft:\t", result_icm_hard_soft)
            else:
                print("Not valid format for hard-soft evaluation")    
                
            #soft vs soft evaluation
            succes, pred_df,gold_df= self.prepare_data_soft_soft()
            if succes: 
                icm_soft = ICM_Soft(pred_df, gold_df, MONO_LABEL_TASK, self.TASK_1_HIERARCHY)
                result_icm_soft_soft=icm_soft.evaluate()
                print("TASK 1 - Result ICM evaluation soft-soft:\t", result_icm_soft_soft)
            else:
                print("Not valid format for soft-soft evaluation")                             
                
            
        if self.task==self.TASK2_TAG:
            #hard vs hard evaluation
            succes, pred_df,gold_df= self.prepare_data_hard_hard()     
            if succes:       
                icm_hard= ICM_Hard(pred_df, gold_df, MONO_LABEL_TASK, self.TASK_2_HIERARCHY)
                result_icm_hard_hard= icm_hard.evaluate()
                print("TASK 2 - Result ICM evaluation hard-hard:\t", result_icm_hard_hard)
                
                fmeasure= FMeasure(pred_df, gold_df, MONO_LABEL_TASK)
                results = fmeasure.evaluate()
                for r in results:
                    print("TASK 2 - Result FMeasure evaluation hard-hard for class: ", r, "=\t", results[r])                
            else:
                print("Not valid format for hard-hard evaluation")            
            
            #hard vs soft evaluation
            succes, pred_df,gold_df= self.prepare_data_hard_soft()
            if succes: 
                icm_soft = ICM_Soft(pred_df, gold_df, MONO_LABEL_TASK, self.TASK_2_HIERARCHY)
                result_icm_hard_soft=icm_soft.evaluate()
                print("TASK 2 - Result ICM evaluation hard-soft:\t", result_icm_hard_soft)
            else:
                print("Not valid format for hard-soft evaluation")    
                
            #soft vs soft evaluation
            succes, pred_df,gold_df= self.prepare_data_soft_soft()
            if succes: 
                icm_soft = ICM_Soft(pred_df, gold_df, MONO_LABEL_TASK, self.TASK_2_HIERARCHY)
                result_icm_soft_soft=icm_soft.evaluate()
                print("TASK 2 - Result ICM evaluation soft-soft:\t", result_icm_soft_soft)
            else:
                print("Not valid format for soft-soft evaluation")                           
            
        
        if self.task==self.TASK3_TAG:
            #hard vs hard evaluation
            succes, pred_df,gold_df= self.prepare_data_hard_hard()     
            if succes:       
                icm_hard= ICM_Hard(pred_df, gold_df, MULTI_LABEL_TASK, self.TASK_3_HIERARCHY)
                result_icm_hard_hard= icm_hard.evaluate()
                print("TASK 3 - Result ICM evaluation hard-hard:\t", result_icm_hard_hard)
                
                fmeasure= FMeasure(pred_df, gold_df, MULTI_LABEL_TASK)
                results = fmeasure.evaluate()
                for r in results:
                    print("TASK 3 - Result FMeasure evaluation hard-hard for class: ", r, "=\t", results[r])                
            else:
                print("Not valid format for hard-hard evaluation")              
            
            #hard vs soft evaluation
            succes, pred_df,gold_df= self.prepare_data_hard_soft()
            if succes: 
                icm_soft = ICM_Soft(pred_df, gold_df, MULTI_LABEL_TASK, self.TASK_3_HIERARCHY)
                result_icm_hard_soft=icm_soft.evaluate()
                print("TASK 3 - Result ICM evaluation hard-soft:\t", result_icm_hard_soft)
            else:
                print("Not valid format for hard-soft evaluation")    
                
            #soft vs soft evaluation
            succes, pred_df,gold_df= self.prepare_data_soft_soft()
            if succes: 
                icm_soft = ICM_Soft(pred_df, gold_df, MULTI_LABEL_TASK, self.TASK_3_HIERARCHY)
                result_icm_soft_soft=icm_soft.evaluate()
                print("TASK 3 - Result ICM evaluation soft-soft:\t", result_icm_soft_soft)
            else:
                print("Not valid format for soft-soft evaluation")
                
            ##### ANGEL ADD ####
        return result_icm_soft_soft
            
                        
    def prepare_data_hard_hard(self):      
        #Generate df predictions
        pred_df = pd.DataFrame.from_dict(self.pred_dict)
        pred_df = pred_df.transpose()
        pred_df.reset_index(inplace=True)
        pred_columns= pred_df.columns.tolist()
        #Check if there is info for hard-hard evaluation
        if (not self.HARD_LABEL_TAG in pred_columns):
            return False, None, None
                
        pred_df = pred_df.rename(columns = {'index':ID, self.HARD_LABEL_TAG:VALUE})
        if self.SOFT_LABEL_TAG in pred_columns:
            pred_df = pred_df.drop(self.SOFT_LABEL_TAG, axis=1)

        
        #Generate df gold
        if self.gold_hard_dict!=None:
            gold_df = pd.DataFrame.from_dict(self.gold_hard_dict)
        else:
            gold_df = pd.DataFrame.from_dict(self.gold_dict)
        gold_df = gold_df.transpose()
        gold_df.reset_index(inplace=True)
        gold_columns= gold_df.columns.tolist()
        #Check if there is info for hard-hard evaluation
        if (not self.HARD_LABEL_TAG in gold_columns):
            return False, None, None
                
        gold_df = gold_df.rename(columns = {'index':ID, self.HARD_LABEL_TAG:VALUE})
        if self.SOFT_LABEL_TAG in gold_columns:        
            gold_df = gold_df.drop(self.SOFT_LABEL_TAG, axis=1)

        return True, pred_df, gold_df       
    
    
    def prepare_data_hard_soft(self):       
        #Generate df predictions
        pred_df = pd.DataFrame.from_dict(self.pred_dict)
        pred_df = pred_df.transpose()
        pred_df.reset_index(inplace=True)        
        pred_columns= pred_df.columns.tolist()
        #Check if there is info for hard-hard evaluation
        if (not self.HARD_LABEL_TAG in pred_columns):
            return False, None, None       

        pred_df = pred_df.rename(columns = {'index':ID, self.HARD_LABEL_TAG:VALUE})        
        if self.SOFT_LABEL_TAG in pred_columns:        
            pred_df = pred_df.drop(self.SOFT_LABEL_TAG, axis=1)          

        pred_df[VALUE]  = pred_df[VALUE].apply(lambda value: self.format_value_hard_soft_mono_label(value))

        
        #Generate df gold
        gold_df = pd.DataFrame.from_dict(self.gold_dict)
        gold_df = gold_df.transpose()
        gold_df.reset_index(inplace=True)        
        gold_columns= gold_df.columns.tolist()
        #Check if there is info for hard-hard evaluation
        if (not self.SOFT_LABEL_TAG in gold_columns):
            return False, None, None                 

        gold_df = gold_df.rename(columns = {'index':ID, self.SOFT_LABEL_TAG:VALUE})       
        if self.HARD_LABEL_TAG in gold_columns:
            gold_df = gold_df.drop(self.HARD_LABEL_TAG, axis=1)

        return True, pred_df, gold_df 
    
        
    def format_value_hard_soft_mono_label(self, value):
        if self.task==self.TASK1_TAG or self.task==self.TASK2_TAG:
            return {value:1.0}
        elif self.task==self.TASK3_TAG:
            soft = dict()
            for c in value:
                soft[c]=1.0
            return soft
                                  
        
    def prepare_data_soft_soft(self):     
        #Generate df predictions
        pred_df = pd.DataFrame.from_dict(self.pred_dict)
        pred_df = pred_df.transpose()
        pred_df.reset_index(inplace=True)        
        pred_columns= pred_df.columns.tolist()
        #Check if there is info for hard-hard evaluation
        if (not self.SOFT_LABEL_TAG in pred_columns):
            return False, None, None                   

        pred_df = pred_df.rename(columns = {'index':ID, self.SOFT_LABEL_TAG:VALUE})
        if self.HARD_LABEL_TAG in pred_columns:
            pred_df = pred_df.drop(self.HARD_LABEL_TAG, axis=1)

        
        #Generate df gold
        gold_df = pd.DataFrame.from_dict(self.gold_dict)
        gold_df = gold_df.transpose()
        gold_df.reset_index(inplace=True)        
        gold_columns= gold_df.columns.tolist()
        #Check if there is info for hard-hard evaluation
        if (not self.SOFT_LABEL_TAG in gold_columns):
            return False, None, None                 

        gold_df = gold_df.rename(columns = {'index':ID, self.SOFT_LABEL_TAG:VALUE})
        if self.HARD_LABEL_TAG in gold_columns:        
            gold_df = gold_df.drop(self.HARD_LABEL_TAG, axis=1)

        return True, pred_df, gold_df                  


def main(argv):
    pred_file = ''
    gold_file = ''
    gold_file_hard = ''
    task=""
    opts, args = getopt.getopt(argv,"hp:g:e:t:",["pfile=","gfile=","efile=","task="])
    for opt, arg in opts:
        if opt == '-h':
            print ('test.py -p <prediction_file> -g <gold_file> -e <gold_hard_file> -t task_name')
            sys.exit()
        elif opt in ("-p", "--pfile"):
            pred_file = arg.strip()
        elif opt in ("-g", "--gfile"):
            gold_file = arg.strip()  
        elif opt in ("-e", "--efile"):
            gold_file_hard = arg.strip() 
        elif opt in ("-t", "--task"):
            task = arg.strip()                     
    print ('Prediction file is ', pred_file)
    print ('Gold file is ', gold_file)
    print ('Gold file hard is ', gold_file_hard)
    print ('Task for evaluation is ', task)
    
    #check if the files exists
    if not check_file_exist(pred_file):
        print("The predictions file does not exist or is empty")
        return
    if not check_file_exist(gold_file):
        print("The gold file does not exist or is empty")
        return
    if gold_file_hard!='':
        if not check_file_exist(gold_file_hard):
            print("The gold hard file does not exist or is empty")
            return
    
    exist2023_evaluation = EXIST_2023_evaluation(pred_file, gold_file, gold_file_hard, task)
    exist2023_evaluation.evaluate()
    
def check_file_exist(path_file):
    path = p.Path(path_file)
    if path.exists():
        if path.stat().st_size > 0:
            return True
        else:
            return False                
    else:
        return False
           
        
if __name__ == '__main__':  
    args = sys.argv[1:]
    _ = main(args)
    

