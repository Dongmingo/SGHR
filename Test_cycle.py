import argparse
import numpy as np
from tqdm import tqdm
from utils.T_eval import *
from utils.r_eval import compute_R_diff
from utils.utils import make_non_exists_dir
from dataops.dataset import get_dataset_name
from TransSync.p2p_reg import name2estimator
from TransSync.Laplacian_TS import pair2globalT_cycle

import open3d
import matplotlib.pyplot as plt

import os
os.environ["DISPLAY"] = ":1"

class cycle_tester():
    def __init__(self, cfg):
        self.cfg = cfg
        self.datasets = get_dataset_name(self.cfg.dataset, './data')
        self.datasetsLo = get_dataset_name('3dLomatch', './data')
        self.estimator = name2estimator[args.estimator](self.cfg)
        self.prelog_dir = f'{self.cfg.save_dir}/cycle_results/cycle_prelogs'
        self.prepose_dir = f'{self.cfg.save_dir}/cycle_results/pcposes'
        make_non_exists_dir(self.prelog_dir)
        make_non_exists_dir(self.prepose_dir)
        self.all_Tscans_topc0 = {}
        #for RR
        self.tau_2 = self.cfg.tau_2
        self.nonconsecutive = True
    
    def savepose(self, dataset, poses):
        make_non_exists_dir(f'{self.prepose_dir}/{dataset.name}')
        poses = poses.reshape(-1,4)
        np.savetxt(f'{self.prepose_dir}/{dataset.name}/pose.txt', poses, delimiter=',')
        
    def savepcds(self, dataset, pcds):
        make_non_exists_dir(f'{self.prepose_dir}/{dataset.name}')
        open3d.io.write_point_cloud(f'{self.prepose_dir}/{dataset.name}/recon.ply', pcds)
        
    def loadpose(self, dataset):
        poses = np.loadtxt(f'{self.prepose_dir}/{dataset.name}/pose.txt',delimiter=',')
        return poses.reshape((-1,4,4))
    
    def savelog(self, dataset, trans):
        make_non_exists_dir(f'{self.prelog_dir}/{dataset.name}')
        writer=open(f'{self.prelog_dir}/{dataset.name}/pre.log','w')
        pair_num=int(len(dataset.pc_ids))
        for i, pair in enumerate(dataset.pair_ids):
            pc0,pc1=pair
            transform_pr=trans[i]
            writer.write(f'{int(pc0)}\t{int(pc1)}\t{pair_num}\n')
            writer.write(f'{transform_pr[0][0]}\t{transform_pr[0][1]}\t{transform_pr[0][2]}\t{transform_pr[0][3]}\n')
            writer.write(f'{transform_pr[1][0]}\t{transform_pr[1][1]}\t{transform_pr[1][2]}\t{transform_pr[1][3]}\n')
            writer.write(f'{transform_pr[2][0]}\t{transform_pr[2][1]}\t{transform_pr[2][2]}\t{transform_pr[2][3]}\n')
            writer.write(f'{0.0}\t{0.0}\t{0.0}\t{1.0}\n')
        writer.close()
    
    def official_RR(self,dataset):
        if not self.cfg.rr: return -1.0
        gt_dir_loc=str.rfind(dataset.gt_dir,'.')
        gt_dir=dataset.gt_dir[0:gt_dir_loc]
        gt_pairs, gt_traj = read_trajectory(f'{gt_dir}.log')
        n_fragments, gt_traj_cov = read_trajectory_info(f'{gt_dir}.info')
        
        pre_dir = f'{self.prelog_dir}/{dataset.name}'
        est_pairs, est_traj = read_pre_trajectory(os.path.join(pre_dir,'pre.log'))
        
        temp_precision, temp_recall,c_flag,c_error = evaluate_registration(n_fragments, est_traj, est_pairs, gt_pairs, gt_traj, gt_traj_cov,err2=self.tau_2,nonconsecutive=self.nonconsecutive)
        return temp_recall
    
    def ecdf(self, dataset, 
             r_splits = [3,5,10,30,45], 
             t_splits = [0.05,0.1,0.25,0.5,0.75]):

        if not self.cfg.ecdf: return np.array([-1.0]),-0.1,-0.1,np.array([-0.1]),-0.1,-0.1
        n_good = 0
        pre_dir = f'{self.prelog_dir}/{dataset.name}'
        est_pairs, est_traj = read_pre_trajectory(os.path.join(pre_dir,'pre.log'))
        Rdiffs, tdiffs = [], []
        for i,pair in enumerate(est_pairs):
            id0,id1,_=pair
            gt = dataset.get_transform(id0,id1)
            est = est_traj[i]
            n_good+=1
            Rdiff = compute_R_diff(gt[0:3,0:3], est[0:3,0:3])
            tdiff = np.sqrt(np.sum(np.square(gt[0:3,-1]-est[0:3,-1]))+1e-8)
            Rdiffs.append(Rdiff)
            tdiffs.append(tdiff)
        Rdiffs = np.array(Rdiffs)
        tdiffs = np.array(tdiffs)
        # rotations
        ecdf_r = [np.mean(Rdiffs<rthes) for rthes in r_splits]
        mean_r = np.mean(Rdiffs)
        med_r = np.median(Rdiffs)
        # translations
        ecdf_t = [np.mean(tdiffs<tthes) for tthes in t_splits]
        mean_t = np.mean(tdiffs)
        med_t = np.median(tdiffs)
        return np.array(ecdf_r), mean_r, med_r, np.array(ecdf_t), mean_t, med_t
    
    def construct_LSW(self, dataset):
        # use predicted overlap ratio
        scoremat = np.load(f'pre/predict_overlap/{dataset.name}/ratio.npy')
        if args.use_gt_overlap_graph:
            scoremat = np.load(f'pre/gt_overlap/{dataset.name}/ratio.npy')
        if args.use_full_graph:
            scoremat =  (np.ones_like(scoremat) - np.eye(scoremat.shape[0])).astype(np.float32)
        n,_ = scoremat.shape     
        # keep symmetry
        for i in range(n):
            scoremat[i,i] = 0
            for j in range(i+1,n):
                scoremat[j,i] = scoremat[i,j]
        # conduct top-k mask
        if args.use_full_graph:
            mask = scoremat
        else:
            mask = np.zeros([n,n])
            for i in range(n):
                score_scan = scoremat[i]
                argsort = np.argsort(-score_scan)[:args.topk]
                mask[i,argsort] = 1
        return scoremat, mask.astype(np.float32)
    
    def get_gt_overlap(self, dataset):
        scoremat = np.load(f'pre/predict_overlap/{dataset.name}/ratio.npy')
        mask = (np.ones_like(scoremat) - np.eye(scoremat.shape[0])).astype(np.float32)        
        # whether we use the ground truth overlap matrix
        # pairwise registration
        n = len(dataset.pc_ids)
        Ts = np.zeros([n,n,4,4])
        overlap = np.zeros([n,n])
        for i in range(n):
            for j in range(n):
                if mask[i,j]>0:
                    if i >= j:continue
                    if np.sum(np.abs(Ts[i,j,0:3,0:3]))<0.001:                    
                        overlap[i,j] = self.estimator.cal_gt_coarse_overlap(dataset, i, j)
                        overlap[j,i] = overlap[i,j]
        return overlap
    
    def onegraph(self, dataset):
        scoremat, mask = self.construct_LSW(dataset)        
        # whether we use the ground truth overlap matrix
        # pairwise registration
        n = len(dataset.pc_ids)
        Ts = np.zeros([n,n,4,4])
        irs = np.zeros([n,n])
        weights = np.zeros([n,n])
        N_pair = 0
        for i in range(n):
            for j in range(i, n):
                if mask[i,j]>0:
                    if i == j:continue
                    # in the following, we must construct a symmetric matrix (weights(if add the noise matrix should also be), Ts) 
                    # for the spectral relaxation solution of rotation synchronization
                    weights[i,j] = 1     
                    weights[j,i] = 1    
                    # If we haven't load the trans and the inv trans, we load the pairwise transformation
                    if np.sum(np.abs(Ts[i,j,0:3,0:3]))<0.001:                    
                        Tij, ir, n_matches = self.estimator.run(dataset, i, j, args.use_gt_coarse_corres, args.use_gt_fine_corres)
                        if n_matches != 0:
                            # guarantee meaningful rotation matrix
                            if np.linalg.det(Tij[0:3,0:3])<0:
                                Tij[0:2] = Tij[[1,0]]   # permute rotation matrix to 1,0,2
                            # we use ransac's inlier number/100
                            irs[i,j], irs[j,i] = ir*n_matches/100, ir*n_matches/100
                            Ts[i,j] = Tij
                            Ts[j,i] = np.linalg.inv(Tij)
                            N_pair += 1
        print(f'Estimate {N_pair} pairs')
        # conduct the global transformation syn
        Tglobals,weights_out = pair2globalT_cycle(weights*scoremat*irs, Ts, args.N_cyclegraph)        
        # save the predicted absolute poses
        self.savepose(dataset, Tglobals)
        return N_pair, Tglobals, ir
    
    def dataseteval(self,dataset,Tpres):
        Tpairs = []
        for pair in dataset.pair_ids:
            id0,id1 = pair
            id0,id1 = int(id0), int(id1)
            T0 = Tpres[id0]
            T1 = Tpres[id1]
            T = np.linalg.inv(T0)@T1
            Tpairs.append(T[None,:,:])
        Tpairs = np.concatenate(Tpairs, axis = 0)
        self.savelog(dataset, Tpairs)
        return Tpairs
        
    def save_gt_overlap(self, gt_overlap, dataset, name):
        savedir = f'pre/gt_overlap/{dataset.name}/{name}'
        make_non_exists_dir(savedir)
        np.save(f'{savedir}/ratio.npy', gt_overlap)
    
    def get_colors(self, N, colormap='viridis'):
        cmap = plt.colormaps.get_cmap(colormap)
        colors = cmap(np.linspace(0, 1, N))[:, :3]
        return colors
    
    def vis_onedatasets(self, datasets):
        for name, dataset in tqdm(self.datasets.items()):
            if type(dataset) is str: continue
            Tpcs = self.loadpose(dataset)
            pcds = []
            N = len(Tpcs)
            save_pcd = open3d.geometry.PointCloud()
            colors = self.get_colors(N, 'viridis')
            for i, Ti in enumerate(Tpcs):
                pcd = dataset.get_pc_o3d(i)
                d_pcd = pcd.voxel_down_sample(0.025)
                abs_Ti = Ti
                d_pcd.transform(abs_Ti)
                d_pcd.paint_uniform_color(colors[i])
                pcds.append(d_pcd)
                save_pcd += d_pcd
            # open3d.visualization.draw_geometries(pcds)
            self.savepcds(dataset, save_pcd)
            
    def run_get_gt_onedatasets(self, datasets):
        for name, dataset in tqdm(self.datasets.items()):
            if type(dataset) is str: continue
            # graph construct and calculate
            gt_overlap = self.get_gt_overlap(dataset)
            self.save_gt_overlap(gt_overlap, dataset, name)
            
    def run_onedatasets(self, datasets):
        Total_N_pair = 0
        RR = []
        IRs = []
        ecdf_rs, mean_rs, med_rs, ecdf_ts, mean_ts, med_ts=[],[],[],[],[],[]
        setname = datasets['wholesetname']
        for name, dataset in tqdm(self.datasets.items()):
            if type(dataset) is str: continue
            if name == 'scene0197_01': continue
            # graph construct and calculate
            N_pair, Tpcs, ir = self.onegraph(dataset)
            IRs.append(ir)
            Total_N_pair += N_pair
            self.all_Tscans_topc0[name] = Tpcs
            # pairtrans cal and save log
            _ = self.dataseteval(datasets[name], Tpcs)
            # official RR calculation
            rr_one = self.official_RR(datasets[name])
            ecdf_r, mean_r, med_r, ecdf_t, mean_t, med_t = self.ecdf(datasets[name])
            RR.append(rr_one)
            ecdf_rs.append(ecdf_r[None])
            mean_rs.append(mean_r)
            med_rs.append(med_r)
            ecdf_ts.append(ecdf_t[None])
            mean_ts.append(mean_t)
            med_ts.append(med_t)
            if self.cfg.rr:
                print(f'{datasets[name].name} rr: {rr_one}')
            if self.cfg.ecdf:
                print(f'{datasets[name].name} ecdf_r:{ecdf_r}, ecdf_t:{ecdf_t}')
        print(f'Conduct {Total_N_pair} pairwise registrations in total.')
        print('IR of the selected pairwise transformations:',np.mean(np.array(IRs)))
        if self.cfg.rr:
            print(f'RR of {setname} - Avg: ', np.mean(np.array(RR)))
        if self.cfg.ecdf:
            ecdf_rs = np.concatenate(ecdf_rs, axis=0)
            ecdf_ts = np.concatenate(ecdf_ts, axis=0)
            ecdf_rs = np.mean(ecdf_rs, axis = 0)
            ecdf_ts = np.mean(ecdf_ts, axis = 0)
            print(f'ECDF_R of {setname} - Statistic: ', ecdf_rs)
            print(f'ECDF_R of {setname} - Mean/Med: ', np.mean(np.array(mean_rs)),'/',np.mean(np.array(med_rs)))
            print(f'ECDF_T of {setname} - Statistic: ', ecdf_ts)
            print(f'ECDF_T of {setname} - Mean/Med: ', np.mean(np.array(mean_ts)),'/',np.mean(np.array(med_ts)))
            
    def run_onedatasets_givengraph(self, datasets):
        RR = []
        setname = datasets['wholesetname']
        for name, dataset in tqdm(self.datasets.items()):
            if type(dataset) is str: continue
            # pairtrans cal and save log
            _ = self.dataseteval(datasets[name], self.all_Tscans_topc0[name])
            # official RR calculation
            rr_one = self.official_RR(datasets[name])
            RR.append(rr_one)
            print(f'{datasets[name].name} rr: {rr_one}')
        print(f'RR of {setname} - Avg: ', np.mean(np.array(RR)))
    
    def visualize_recon(self):
        self.vis_onedatasets(self.datasets) 
    
    def run_get_gt(self):
        self.run_get_gt_onedatasets(self.datasets) 
        
    def run(self):
        self.run_onedatasets(self.datasets) 
        if self.datasets['wholesetname'] == '3dmatch':
            self.run_onedatasets_givengraph(self.datasetsLo)

            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',         default='3dmatch',  type=str,   help='name of dataset')
    
    # Customized parameters
    parser.add_argument('--calculate_coarse_overlap', action='store_true')
    parser.add_argument('--visualize_reconstruction', action='store_true')
    parser.add_argument('--use_full_graph', action='store_true')
    parser.add_argument('--use_gt_overlap_graph', action='store_true')
    parser.add_argument('--use_gt_coarse_corres', action='store_true')
    parser.add_argument('--use_gt_fine_corres', action='store_true')
    
    # for multireg algorithm
    parser.add_argument('--estimator',       default='yoho',    type=str,     help='name of estimator')
    parser.add_argument('--topk',            default=10,        type=int,     help='the top k overlapping scans used for transformation syn')
    parser.add_argument('--N_cyclegraph',    default=50,        type=int,     help='we WLS the graph for 100 iterations')
    
    # for evaluation
    parser.add_argument('--save_dir',        default='./pre',   type=str,     help='for eval results')
    parser.add_argument('--inlierd',         default=0.07,      type=float,   help='inlier threshold for RANSAC')
    parser.add_argument('--rr',              action='store_true')
    parser.add_argument('--tr',              action='store_true')
    parser.add_argument('--ecdf',            action='store_true')
    
    parser.add_argument('--tau_2',           default=0.2,    type=float,   help='Thres for RR')
    
    
    args = parser.parse_args()
    
    tester = cycle_tester(args)
    if args.calculate_coarse_overlap:
        tester.run_get_gt()
    elif args.visualize_reconstruction:
        tester.visualize_recon()
    else:
        tester.run()