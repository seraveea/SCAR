# including the design of retrieval router
import numpy as np
import numpy as np
import scipy.spatial.distance as dist
import bisect



import bisect
import numpy as np

class SCAR:
    def __init__(self, alpha=0.9,
                    k_neighbors=10,
                    priority=(2, 0, 1),         
                    sim_mode="S_cos",            
                    sim_threshold=0.70,          
                    overlap_down=0.2,            
                    downweight_pair=0.80,        
                    weight_floor=0.08,           
                    k_sim=20,                   
                    max_nodes_for_sim=256,      
                    seed=0
                    ):
            self.alpha = float(alpha)
            self.k = int(k_neighbors)
            self.priority = priority
            self.sim_mode = sim_mode
            self.sim_threshold = sim_threshold
            self.overlap_down = overlap_down
            self.downweight_pair = downweight_pair
            self.weight_floor = weight_floor
            self.k_sim = k_sim
            self.max_nodes_for_sim = max_nodes_for_sim
            self.seed = seed

    def align_data(self, raw_data_list):
        """Union Set"""
        all_ids = set()
        for res in raw_data_list:
            all_ids.update(res['ids'])
        
        unique_ids = sorted(list(all_ids))
        id_to_idx = {uid: i for i, uid in enumerate(unique_ids)}
        idx_to_id = {i: uid for uid, i in id_to_idx.items()}
        N = len(unique_ids)
        aligned_vecs_list = []
        global_scores = np.zeros(N)
        for res in raw_data_list:
            ids = res['ids']
            scores = self.minmax(res['scores'])
            vecs = np.array(res['vecs']) 
            dim = vecs.shape[1]
            
            full_matrix = np.zeros((N, dim))
            
            for i, entity_id in enumerate(ids):
                idx = id_to_idx[entity_id]
                full_matrix[idx] = vecs[i]
                # 累加 CombMNZ 分数作为 MR 种子
                global_scores[idx] += scores[i]
            aligned_vecs_list.append(full_matrix)
        return aligned_vecs_list, global_scores, idx_to_id

    def build_affinity_matrix(self, vecs):
        """KNN + Gaussian Kernel"""
        norm = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10
        vecs_norm = vecs / norm
        dists = dist.squareform(dist.pdist(vecs_norm, metric='euclidean'))
        sigma = np.mean(dists) + 1e-10
        W = np.exp(- (dists ** 2) / (2 * sigma ** 2))
        
        raw_norms = np.linalg.norm(vecs, axis=1)
        zero_mask = raw_norms < 1e-6
        W[zero_mask, :] = 0
        W[:, zero_mask] = 0
        
        N = W.shape[0]
        k_real = min(self.k, N - 1)
        if k_real > 0:
            mask = np.zeros_like(W)
            for i in range(N):
                top_k_idx = np.argsort(W[i])[-k_real:]
                mask[i, top_k_idx] = 1

            # mask = mask * mask.T
            W = W * mask
        
        # S = D^-0.5 * W * D^-0.5
        row_sum = np.sum(W, axis=1) + 1e-10
        D_inv_sqrt = np.power(row_sum, -0.5)
        S = np.diag(D_inv_sqrt) @ W @ np.diag(D_inv_sqrt)
        return S
    
    def run_mr(self, S, initial_scores):
        N = S.shape[0]
        I = np.eye(N)
        try:
            # F = (I - alpha * S)^-1 * Y
            inv_matrix = np.linalg.inv(I - self.alpha * S)
            final_scores = np.dot(inv_matrix, initial_scores)
        except np.linalg.LinAlgError:
            return initial_scores
        return final_scores

    def run(self,raw_data_list):
        self.rng = np.random.default_rng(self.seed)

        aligned_vecs_list, combmnz_scores, idx_to_id = self.align_data(raw_data_list)
        num_views = len(aligned_vecs_list)
        N = len(idx_to_id)

        view_exist = []
        for i in range(num_views):
            vecs = aligned_vecs_list[i]
            view_exist.append(np.linalg.norm(vecs, axis=1) > 1e-6)

        S_list = []
        mr_results = []
        for i in range(num_views):
            vecs = aligned_vecs_list[i]
            if np.all(vecs == 0):
                S_list.append(None)
                mr_results.append(combmnz_scores.copy())
                continue

            S = self.build_affinity_matrix(vecs)   # (N,N) dense
            S_list.append(S)

            refined = self.run_mr(S, combmnz_scores)
            mr_results.append(refined)

        static_weights = np.full(num_views, 1.0 / num_views, dtype=np.float32)
        final_weights = static_weights

        # priority rank
        pr_rank = {int(v): r for r, v in enumerate(self.priority)}
        for i in range(num_views):
            if i not in pr_rank:
                pr_rank[i] = len(pr_rank) + i

        # local scale is a matrix [num_views, num_entities]
        local_scale = [np.ones(N, dtype=np.float32) for _ in range(num_views)]
        weight_scale = np.ones(num_views, dtype=np.float32)

        for i in range(num_views):
            for j in range(i + 1, num_views):
                if S_list[i] is None or S_list[j] is None:
                    continue

                if self.sim_mode == "rankcorr":
                    sim = self._rankcorr_similarity(S_list[i], view_exist[i], S_list[j], view_exist[j], self.k_sim)
                elif self.sim_mode == "S_cos":
                    sim = self._S_cos_similarity(S_list[i], view_exist[i], S_list[j], view_exist[j])
                else:
                    raise ValueError(f"Unknown sim_mode: {self.sim_mode}")

                if sim < self.sim_threshold:
                    continue

                hi, lo = (i, j) if pr_rank[i] < pr_rank[j] else (j, i)

                weight_scale[lo] *= float(self.downweight_pair)

                overlap = view_exist[hi] & view_exist[lo]
                local_scale[lo][overlap] *= float(self.overlap_down)

        # weight_scale + floor + renorm
        final_weights = final_weights * weight_scale
        final_weights = np.maximum(final_weights, float(self.weight_floor))
        final_weights = final_weights / (final_weights.sum() + 1e-12)

        weighted_scores_list = []
        for i, scores in enumerate(mr_results):
            mu = float(np.mean(scores))
            sigma = float(np.std(scores)) + 1e-9
            z = (scores - mu) / sigma
            z = z * local_scale[i]
            weighted_scores_list.append(z * float(final_weights[i]))
        final_scores = np.sum(np.stack(weighted_scores_list, axis=0), axis=0)
        sorted_indices = np.argsort(final_scores)[::-1]
        ranked_results = [(idx_to_id[idx], float(final_scores[idx])) for idx in sorted_indices]
        return ranked_results    

    # -------- helper A: top-k neighbors from S row (exclude self) --------
    def _topk_from_S(self, S, nodes, k):
        # returns dict[u] -> (nbr_ids, nbr_ranks) with ranks 1..k
        out = {}
        k_eff = min(k, S.shape[0] - 1)
        for u in nodes:
            row = S[u].copy()
            row[u] = -1e9
            # take top-k_eff
            idx = np.argpartition(row, -k_eff)[-k_eff:]
            idx = idx[np.argsort(row[idx])[::-1]]
            out[int(u)] = (idx.astype(np.int32), np.arange(1, idx.size + 1, dtype=np.int32))
        return out

    # -------- helper B: rank correlation similarity (Spearman-like on ranks with missing=k+1) --------
    def _rankcorr_similarity(self, Sa, mask_a, Sb, mask_b, k):
        both = mask_a & mask_b
        nodes = np.flatnonzero(both)
        if nodes.size == 0:
            return 0.0

        if self.max_nodes_for_sim and nodes.size > self.max_nodes_for_sim:
            nodes = self.rng.choice(nodes, size=self.max_nodes_for_sim, replace=False)

        k_eff = min(k, Sa.shape[0] - 1)
        na = self._topk_from_S(Sa, nodes, k_eff)
        nb = self._topk_from_S(Sb, nodes, k_eff)
        sims = []
        missing = k_eff + 1

        for u in nodes:
            a_ids, a_r = na[int(u)]
            b_ids, b_r = nb[int(u)]

            da = {int(x): int(r) for x, r in zip(a_ids, a_r)}
            db = {int(x): int(r) for x, r in zip(b_ids, b_r)}

            uni = list(set(da.keys()) | set(db.keys()))
            if len(uni) <= 1:
                continue

            ra = np.array([da.get(v, missing) for v in uni], dtype=np.float32)
            rb = np.array([db.get(v, missing) for v in uni], dtype=np.float32)

            # Pearson on ranks (Spearman-like since these are ranks)
            ra = ra - ra.mean()
            rb = rb - rb.mean()
            denom = (np.linalg.norm(ra) * np.linalg.norm(rb) + 1e-12)
            sims.append(float((ra @ rb) / denom))

        if len(sims) == 0:
            return 0.0
        return float(np.mean(sims))

    # -------- helper C: S-submatrix cosine/Frobenius similarity --------
    def _S_cos_similarity(self, Sa, mask_a, Sb, mask_b):
        both = mask_a & mask_b
        nodes = np.flatnonzero(both)
        if nodes.size == 0:
            return 0.0

        if self.max_nodes_for_sim and nodes.size > self.max_nodes_for_sim:
            nodes = self.rng.choice(nodes, size=self.max_nodes_for_sim, replace=False)

        A = Sa[np.ix_(nodes, nodes)].astype(np.float32, copy=False)
        B = Sb[np.ix_(nodes, nodes)].astype(np.float32, copy=False)

        # Frobenius cosine similarity
        num = float(np.sum(A * B))
        den = float(np.linalg.norm(A) * np.linalg.norm(B) + 1e-12)
        return num / den
    
    @staticmethod
    def minmax(scores):
        scores = np.array(scores, dtype=float)
        s_min = scores.min()
        s_max = scores.max()
        if s_max - s_min < 1e-12:
            return np.zeros_like(scores)
        return (scores - s_min) / (s_max - s_min)

