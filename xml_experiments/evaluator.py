from xclib.evaluation.xc_metrics import Metrics, compute_inv_propesity
from xclib.data import data_utils as du
from scipy.sparse import csr_matrix
import numpy as np
import warnings
import time
import os

class Evaluator:
    def __init__(self, data_path):
        self.DATA_DIR = data_path
        self.A, self.B = 0.6, 2.6
        
        self.filter_file = self.DATA_DIR + "filter_labels_test.txt"
        if not os.path.exists(self.filter_file): self.filter_file = None
            
        # Read data
        self.load_overlap()

        try:
            self.train_labels_file = self.DATA_DIR + "trn_X_Y.txt"
            self.test_labels_file = self.DATA_DIR + "tst_X_Y.txt"
            self.train_y = du.read_sparse_file(self.train_labels_file, safe_read=False)
            self.test_y = du.read_sparse_file(self.test_labels_file, safe_read=False)
        except:
            _, self.train_y, _, _, _ = du.read_data(self.DATA_DIR + "train.txt")
            _, self.test_y, _, _, _ = du.read_data(self.DATA_DIR + "test.txt")
            
        self.test_y = self.remove_overlap(self.test_y, self.docs, self.lbls)
        self.num_labels = self.train_y.shape[1]
        
        self.docs_train_pos, self.lbls_train_pos = None, None
        if os.path.exists(self.DATA_DIR + "train_raw_ids.txt"): self.get_test_users()
        
    def load_overlap(self):
        self.docs, self.lbls = None, None
        if self.filter_file is None: return
        
        temp = np.loadtxt(self.filter_file, dtype = np.int32)
        self.docs = temp[:, 0]
        self.lbls = temp[:, 1]

    def remove_overlap(self, score_mat, docs, lbs):
        if docs is None or lbs is None: return score_mat
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            score_mat[docs, lbs] = 0.0
            score_mat = score_mat.tocsr()
            score_mat.eliminate_zeros()
        return score_mat

    def get_test_users(self):
        # Make test user --> user id map
        self.test_users = [] ## Final

        pos_map = {}; at = 0
        f = open(self.DATA_DIR + "train_raw_ids.txt", 'r')
        while 1:
            line = f.readline()
            if not line: break
            pos_map[line.strip()] = at
            at += 1
        f.close()

        self.docs_train_pos = []; self.lbls_train_pos = []; at = 0
        f = open(self.DATA_DIR + "test_raw_ids.txt", 'r')
        print("Getting train:test mapping..")
        while 1:
            line = f.readline()
            if not line: break
            train_number = pos_map[line.strip()]

            self.test_users.append(train_number) ## Final

            train_pos = self.train_y[train_number].nonzero()[1]
            for l in train_pos: 
                self.docs_train_pos.append(at)
                self.lbls_train_pos.append(l)
            at += 1
        f.close()
        
    def evaluate(self, score_mat):
        # Make evaluator
        inv_psp = compute_inv_propesity(self.test_y, self.A, self.B)
        evaluator = Metrics(self.test_y, inv_psp = inv_psp)

        # Remove training positives from predictions
        score_mat = self.remove_overlap(score_mat, self.docs_train_pos, self.lbls_train_pos)
        
        # Filter labels which are the same as input
        score_mat = self.remove_overlap(score_mat, self.docs, self.lbls)
    
        p, ndcg, psp, pndcg = evaluator.eval(score_mat, K = 5)
        metrics = self.pretty_print([ 
            [ p, 'Pk' ], 
            [ ndcg, 'Nk' ], 
            [ psp, 'PSPk' ], 
            [ pndcg, 'PSNk' ], 
        ], [ 1, 3, 5 ])
        return metrics
    
    def pretty_print(self, metrics, Ks):
        ret = ""
        for metric in metrics:
            ret += metric[1] + '\t'
            for k in Ks:
                ret += str(k) + ": " + str(round(100.0 * metric[0][k-1], 6)) + '\t'
            ret += '\n'
        return ret
        