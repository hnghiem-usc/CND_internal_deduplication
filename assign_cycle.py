#!/usr/bin/env python
# coding: utf-8

# In[117]:


#INSTALL ALL OF THESE PACKAGES USING ANACONDA OR PIP FIRS
import numpy as np 
import pandas as pd 
import networkx as nx
import csv
import argparse 
from itertools import combinations


# In[132]:


def processExt(path, keyname):
    '''
    Helper function that proceses the origal extract to only keep the internal matches in 1 source only
    @param path: str, path to where the extract is. Best to be in the same directory as this file.
    @keyname: str, name of the record source to restrict the path into.
    '''
    ext = pd.read_csv(path)
    ext[['RECORD_ID1','RECORD_ID2']] = ext[['RECORD_ID1','RECORD_ID2']].astype(int)
    ext = ext.loc[(ext.RECORD2_SOURCE == keyname) & (ext.RECORD1_SOURCE == ext.RECORD2_SOURCE) & (ext.DECISION =='M')][['RECORD_ID1','RECORD_ID2','MATCH_PROBABILITY']].drop_duplicates()
    return ext



class InternalDeduplicator():
    '''
    This class takes the extract produces by ChoiceMaker and uses NetowrkX to locate 
    cycles and assign a single unique ID to each key ID in the extract (NOT THE ENTIRE SET)
    '''
    def __init__(self, extract, keyname:str):
        '''
        Initializer
        @param extract: dataframe, should only contain only internal MATCHES from a single source ONLY
        @param keyname: string, name of the key in the data
        '''
        self.ext = extract[['RECORD_ID1',  'RECORD_ID2','MATCH_PROBABILITY']] #might change to RECORD1_ID1 next iteration
        self.keyname = keyname
        
    def __getCycles(self):
        '''
        Use NetowrkX to return the cycles.
        '''
        ##I. REVERSE THE EDGES' DIRECTION TO CREATE FULL GRAPH
        #Add the back edges by reversing the direction between each pair. 
        self.ext_back = self.ext.copy()
        # ext_back = ext[['RECORD_ID2','RECORD_ID1']]
        self.ext_back = self.ext_back.rename(columns={'RECORD_ID2':'RECORD_ID1','RECORD_ID1':'RECORD_ID2'})
        self.all_edges = pd.concat([self.ext,self.ext_back],axis=0)
        self.all_edges.sort_values('RECORD_ID1', inplace=True)
                     
        ##Use network graph to find all the cycles
        self.G = nx.DiGraph(self.all_edges[['RECORD_ID1','RECORD_ID2']].values.tolist())                      
        self.dup = []
        self.max_len = 1 #initialize with cycle of size 1 (a vertice by itself)

        for cycle in nx.simple_cycles(self.G):
            if len(cycle) > 1:
                self.dup.append(cycle)
                if len(cycle)>self.max_len:
                    self.max_len = len(cycle) #keep track of the max size of cycles
       
        ## CREATE SET OF UNIQUE CYCLES ONLY
        self.dup_set = list(set(frozenset(item) for item in self.dup))
        self.dup_unique = [list(item) for item in self.dup_set]
        
        ## REFORMAT INTO DATAFRAME
        self.dup_df = pd.DataFrame(self.dup_unique, columns=['id_'+str(i+1) for i in range(self.max_len)])
        self.dup_df['num_v'] = self.max_len -  self.dup_df.isnull().sum(axis=1)
        ##Force ID's to string 
        self.dup_df.fillna(0,inplace=True)
        self.dup_df = self.dup_df.astype({'id_1':int, 'id_2':int, 'id_3':int, 'id_4':int,'id_5':int,'id_6':int })
        self.dup_df['dupset'] = self.dup_df.apply(lambda x: frozenset(x[:x.num_v]),axis=1)
        
    def __deleteSmallCycle(self, zformat=6):
        '''
        Give the dataframe of all cycles, return a clean up dataframe in which 
        only the set of all postives are deleted. 
        Ex: biggest cycle of size (1,2,3,4,5). Then all sub-cycles (1,2,3),(2,3,4,5,6) 
        are DELETED if they exist
        @param zformat: int, the length of the unique_ID to be assigned. Default 6.
        '''
        cy_df = self.dup_df
        max_n = cy_df.num_v.max()
        tbd_all = [] 
        for s in range(max_n, 2 , -1):
            for i, r in cy_df.loc[cy_df.num_v == s,:].iterrows():
                for c in combinations(r[:s], s-1):
                    frozen_c = frozenset(c)
                    #Get the index to be deleted
                    tbd = cy_df[cy_df.dupset == frozen_c].index.tolist()
                    tbd_all.append(tbd)
        #                 print(c)

        #             print('-------------')
        self.tbd_all = np.unique(np.array(sum(tbd_all,[])))
        self.clean_dup = cy_df[~cy_df.index.isin(self.tbd_all)].sort_values('num_v')  
#         return tbd_all, cy_df[~cy_df.index.isin(tbd_all)].sort_values('num_v')  
        self.clean_dup['temp'] = np.arange(1,self.clean_dup.shape[0]+1)
        self.clean_dup['unique_id'] = self.clean_dup.temp.apply(lambda x: self.keyname +str(x).zfill(zformat))
        del self.clean_dup['temp']
        
        
    def getCycleReport(self):
        '''
        Print the reports for the cycles found, not yet deduplicated.
        '''
        self.__getCycles()
        self.__deleteSmallCycle()
        print("COUNT OF THE NUMBER CYCLES WITH SIZE NUM_V:\n")
        print(self.clean_dup.groupby('num_v').num_v.count())
        
        
    def __assignID(self):
        '''
        Assign a unique ID to each key in the source using the cycles produced.
        '''
        self.idf = pd.DataFrame(columns=[self.keyname, 'unique_id', 'num_v'])
        index = 0
        for _, r in self.clean_dup.iterrows():
            for item in r.dupset:
                self.idf.loc[index,:] = (item, r.unique_id, r.num_v)
                index+=1
        #     print(i)
        self.idf.drop_duplicates().sort_values('unique_id', inplace=True)
        self.idf.reset_index(inplace=True, drop=True)
        return self.idf

    def getTies(self):
        '''
        Get the original ID that is in 2 seprate cycles.
        '''
        self.dup_id = self.__assignID()
        m = self.dup_id.groupby(self.keyname)[self.keyname].count()
        check = m[m>1].index.tolist()
        self.check_df = self.dup_id[self.dup_id[self.keyname].isin(check)].sort_values(self.keyname)
        self.check_df.sort_values(self.keyname)
        self.check_dup =  self.clean_dup.loc[self.clean_dup.unique_id.isin(self.check_df.unique_id)]
    
    def __lookupWeights(self, group, df):
        '''
        Helper function: given a group as set, and a lookup extract that contains RECORD_ID1, RECORD_ID2
        and MATCH_PROBABILITY, return the AVERAGE weight of that group.
        @group: a frozen set that contains all the vertices in the cycle
        @df: dataframe, that contains all the match pairs with probabiltiies, or the CM extracts
        '''
        w_all = []
        try:
            for g in combinations(group,2):
                ##Look  up the edge weight 
                w1 = df.loc[(df.RECORD_ID1 ==  g[0]) &  (df.RECORD_ID2 == g[1]), 'MATCH_PROBABILITY'].values
                w2 = df.loc[(df.RECORD_ID2 ==  g[0]) &  (df.RECORD_ID1 == g[1]), 'MATCH_PROBABILITY'].values
                w = max(set.union(set(w1),set(w2)))
                #Add to the general weight
        #         print(w_all)
                w_all.append(w)
        except:
            print("ERROR, check g:", g)
        return np.mean(w_all) #Return the avarage of the weights
    
    def getTieBreaker(self):
        '''
        If a keu ID is assigned 2 or more unique IDs, then assign to the one with the highest avarage probability
        '''
        cycle_weights = pd.Series(self.check_dup.apply(lambda x: self.__lookupWeights(x.dupset,self.ext), axis=1), name='cycle_weights')
        self.check_dup = pd.concat([self.check_dup, cycle_weights], axis=1)
        ##Merge to get the final list
        prefinal = pd.merge(left=self.check_df, left_on='unique_id'
        ,right=self.check_dup[['unique_id','cycle_weights']], right_on='unique_id'
        ,how='left').drop_duplicates()
        
        final = prefinal.groupby(self.keyname).max()[['cycle_weights','unique_id','num_v']]
        self.final = final.reset_index()
        
        
    def getFinalUniqueID(self):
        '''
        Implement tie-breaker so all origal ID has only 1 new assign ID.
        '''
        self.key_del = self.final[self.keyname].unique()
        self.dup_id.drop(self.dup_id[self.dup_id[self.keyname].isin(self.key_del)].index, inplace=True)
        self.final_id = pd.concat([self.dup_id, self.final[[self.keyname,'unique_id','num_v']]]).drop_duplicates().reset_index(drop=True)
        return self.final_id
    
    
    def runAll(self):
        '''
        Wrapper function that runs from beginning to end after Initializer.
        For convenience
        '''
        self.__getCycles()
        self.__deleteSmallCycle()
        self.getTies()
        self.getTieBreaker()
        final = self.getFinalUniqueID()
        return final 


# In[102]:


### TEST CLASS

if __name__ == '__main__':
    myparser = argparse.ArgumentParser(description="Accept arguments to run from the command")
    myparser.add_argument("-ext", action="store", dest='ext', help="path to the location of the datafile")
    myparser.add_argument("-key", action="store", dest='key', help="name of the argument")


    # In[131]:


    #Parse the arguments
    args = myparser.parse_args()
    ext_name = str(args.ext)
    keyname  = str(args.key)
    
#     print(ext_name)
#     print(keyname)
    
    extract = processExt(ext_name,  keyname)
#     print(extract.head(3))
    deduper = InternalDeduplicator(extract, keyname)
    final_id = deduper.runAll()
    
    assert final_id[keyname].nunique() == final_id.shape[0]
    final_id.sort_values('unique_id', inplace=True)
    final_id.to_csv(keyname+"_internal_dedup.csv",index=None)

######################## END OF CODE ################################


