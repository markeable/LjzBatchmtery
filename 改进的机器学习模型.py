# -*- coding: utf-8 -*-
"""
【最终方案 - 邻域特征 Res-MLP v6.1 最终修正版】
此版本：
1. 核心修正: 重新补回了缺失的 interpolate_negatives 函数定义。
"""
import os
import time
import joblib
import numpy as np
import pandas as pd
import gc
import ast
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from osgeo import gdal, osr
from scipy.spatial import cKDTree
from sklearn.model_selection import KFold, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.interpolate import griddata
import optuna
from optuna.samplers import TPESampler
import torch
import torch.nn as nn
import copy
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from joblib import Parallel, delayed

# 可选依赖
try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False
try:
    import catboost as cb
    HAS_CATBOOST = True
except Exception:
    HAS_CATBOOST = False
try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False


class CFG:
    RANDOM_STATE = 42
    OPTUNA_TRIALS_BASE = 50
    OPTUNA_TRIALS_META = 50
    CV_SPLITS = 5
    CHUNK_SIZE = 8192
    PIXEL_THRESH_FACTOR = 0.7
    
    MLP_EPOCHS_MAX = 300
    EARLY_STOP = 25
    WEIGHT_DECAY = 5e-4 
    MLP_DROPOUT = 0.4
    BATCH_SIZE = 8
    NUM_WORKERS = 0
    
    NEIGHBORHOOD_SIZE = 5
    ENABLE_SHAP = True
    PREDICT_BATCH_SIZE = 4096

    MLP_LR = 3e-4

class IO:
    @staticmethod
    def validate_image(path_or_ds):
        ds = gdal.Open(path_or_ds) if isinstance(path_or_ds, str) else path_or_ds
        if ds is None: raise FileNotFoundError("无法打开影像")
        gt, proj = ds.GetGeoTransform(), ds.GetProjection()
        ncols, nrows = ds.RasterXSize, ds.RasterYSize; bands = ds.RasterCount
        arr = ds.ReadAsArray().astype(np.float32)
        nodata = ds.GetRasterBand(1).GetNoDataValue() if bands > 0 else None
        return gt, proj, ncols, nrows, np.nan_to_num(arr, nan=1e-6), bands, nodata
    @staticmethod
    def lonlat_to_proj(pts_df, proj_wkt):
        src=osr.SpatialReference(); src.ImportFromEPSG(4326)
        dst=osr.SpatialReference(); dst.ImportFromWkt(proj_wkt)
        tf = osr.CoordinateTransformation(src, dst)
        return np.array([tf.TransformPoint(x, y)[:2] for x, y in zip(pts_df['Longitude'], pts_df['Latitude'])])
    @staticmethod
    def save_tiff(arr, gt, proj, out_dir, name):
        os.makedirs(out_dir, exist_ok=True); path = os.path.join(out_dir, f"{name}.tif")
        rows, cols = arr.shape; drv = gdal.GetDriverByName('GTiff')
        ds = drv.Create(path, cols, rows, 1, gdal.GDT_Float32)
        ds.SetGeoTransform(gt); ds.SetProjection(proj)
        ds.GetRasterBand(1).WriteArray(arr); ds.GetRasterBand(1).SetNoDataValue(-9999); ds.FlushCache()
        return path
    @staticmethod
    def save_csv(data, out_dir, name):
        os.makedirs(out_dir, exist_ok=True); df = pd.DataFrame(data)
        path = os.path.join(out_dir, f"{name}.csv"); df.to_csv(path, index=False, encoding='utf-8-sig')
        return path

def match_points(xy_proj, geotrans, ncols, nrows, lons, lats, depths, out_dir, img_name):
    A = np.array([[geotrans[1], geotrans[2]],[geotrans[4], geotrans[5]]])
    B = np.vstack((xy_proj[:,0]-geotrans[0], xy_proj[:,1]-geotrans[3]))
    try: sol = np.linalg.solve(A, B)
    except Exception: return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    cols = np.round(sol[0]).astype(int); rows = np.round(sol[1]).astype(int)
    mask = (rows>=0)&(rows<nrows)&(cols>=0)&(cols<ncols)
    if not np.any(mask): return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    rows, cols = rows[mask], cols[mask]; depths, lons, lats = depths[mask], lons[mask], lats[mask]
    px = geotrans[0] + (cols+0.5)*geotrans[1] + (rows+0.5)*geotrans[2]
    py = geotrans[3] + (cols+0.5)*geotrans[4] + (rows+0.5)*geotrans[5]
    pixel_centers = np.column_stack((px, py)); dists = np.linalg.norm(xy_proj[mask] - pixel_centers, axis=1)
    pix_w, pix_h = abs(geotrans[1]), abs(geotrans[5]); pix_diag = np.sqrt(pix_w**2 + pix_h**2)
    thresh = pix_diag * CFG.PIXEL_THRESH_FACTOR; good = dists <= thresh
    if not np.any(good): good = dists <= (pix_diag * 1.5)
    final_rows, final_cols, final_depths = rows[good], cols[good], depths[good]
    final_lons, final_lats, final_dists = lons[good], lats[good], dists[good]
    try:
        IO.save_csv({'Longitude': final_lons, 'Latitude': final_lats, 'Depth': final_depths,
                     'row': final_rows, 'col': final_cols, 'dist': final_dists}, out_dir, f"sample_pixel_dist_{img_name}")
        plt.figure(figsize=(6,4)); plt.hist(final_dists, bins=30); plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"dist_hist_{img_name}.png")); plt.close()
    except Exception: pass
    return final_rows, final_cols, final_depths, final_lons, final_lats, final_dists

class Searcher:
    @staticmethod
    def _base_obj(trial, model_name, X, y):
        use_gpu = False
        if model_name == 'RandomForest':
            p = {'n_estimators': trial.suggest_int('n_estimators',50,300),'max_depth':trial.suggest_int('max_depth',3,30),'min_samples_leaf':trial.suggest_int('min_samples_leaf',1,10)}
            model = RandomForestRegressor(**p, n_jobs=1, random_state=CFG.RANDOM_STATE)
        elif model_name == 'SVR':
            p = {'C': trial.suggest_float('C',0.1,100,log=True),'gamma': trial.suggest_categorical('gamma',['scale','auto'])}
            model = SVR(**p, kernel='rbf')
        elif model_name == 'XGBoost' and HAS_XGB:
            p = {'n_estimators': trial.suggest_int('n_estimators',50,300),'learning_rate': trial.suggest_float('learning_rate',1e-3,0.2,log=True),'max_depth':trial.suggest_int('max_depth',3,10),'subsample':trial.suggest_float('subsample',0.6,1.0),'colsample_bytree':trial.suggest_float('colsample_bytree',0.6,1.0)}
            model = xgb.XGBRegressor(**p, n_jobs=1, random_state=CFG.RANDOM_STATE, verbosity=0)
        elif model_name == 'CatBoost' and HAS_CATBOOST:
            p = {'iterations': trial.suggest_int('iterations',100,1000),'learning_rate': trial.suggest_float('learning_rate',1e-3,0.2,log=True),'depth':trial.suggest_int('depth',3,10)}
            model = cb.CatBoostRegressor(**p, thread_count=1, random_seed=CFG.RANDOM_STATE, verbose=0)
        else: return -1e9
        kf = KFold(3, shuffle=True, random_state=CFG.RANDOM_STATE)
        scores = []
        for tr, te in kf.split(X):
            try: model.fit(X[tr], y[tr]); scores.append(r2_score(y[te], model.predict(X[te])))
            except Exception: return -1e9
            trial.report(np.mean(scores), len(scores));
            if trial.should_prune(): raise optuna.exceptions.TrialPruned()
        return np.mean(scores)

    @staticmethod
    def search_base(model_name, X, y, n_trials=CFG.OPTUNA_TRIALS_BASE, n_jobs=-1):
        if (model_name=='XGBoost' and not HAS_XGB) or (model_name=='CatBoost' and not HAS_CATBOOST):
            print(f"跳过{model_name}（缺依赖）"); return {}
        pruner = optuna.pruners.MedianPruner(n_warmup_steps=2)
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=CFG.RANDOM_STATE), pruner=pruner)
        try: study.optimize(lambda t: Searcher._base_obj(t, model_name, X, y), n_trials=n_trials, n_jobs=n_jobs)
        except Exception as e: print(f"{model_name} search error: {e}"); return {}
        return study.best_trial.params if study.best_trial else {}

class BaseTrainer:
    @staticmethod
    def _model_class(name, params):
        use_gpu = False
        if name=='RandomForest': return RandomForestRegressor, {**params, 'n_jobs':-1, 'random_state':CFG.RANDOM_STATE}
        if name=='SVR': return SVR, {**params, 'kernel':'rbf'}
        if name=='XGBoost' and HAS_XGB: return xgb.XGBRegressor, {**params, 'n_jobs':-1, 'random_state':CFG.RANDOM_STATE, 'verbosity':0}
        if name=='CatBoost' and HAS_CATBOOST: return cb.CatBoostRegressor, {**params, 'thread_count':-1, 'random_seed':CFG.RANDOM_STATE, 'verbose':0}
        return None, None
    @staticmethod
    def predict_in_chunks(model, X):
        n = X.shape[0]; preds = np.zeros(n, dtype=np.float32)
        for i in range(0, n, CFG.CHUNK_SIZE): j = min(i+CFG.CHUNK_SIZE, n); preds[i:j] = model.predict(X[i:j])
        return preds
    @staticmethod
    def train_oof(name, X, y, flat_spec, nrows, ncols, params, scaler=None):
        cls, kwargs = BaseTrainer._model_class(name, params)
        if cls is None: return None, None, None
        X_train = scaler.transform(X) if (name=='SVR' and scaler is not None) else X
        flat_train = scaler.transform(flat_spec) if (name=='SVR' and scaler is not None) else flat_spec
        oof_pred = np.zeros(len(y), dtype=np.float32)
        kf = KFold(CFG.CV_SPLITS, shuffle=True, random_state=CFG.RANDOM_STATE)
        for tr, va in kf.split(X_train):
            m = cls(**kwargs); m.fit(X_train[tr], y[tr]); oof_pred[va] = m.predict(X_train[va])
        full_model = cls(**kwargs); full_model.fit(X_train, y)
        full_pred_flat = BaseTrainer.predict_in_chunks(full_model, flat_train)
        full_map_pred = full_pred_flat.reshape(nrows, ncols)
        return oof_pred, full_map_pred, full_model

class MetaHelper:
    @staticmethod
    def eval(y_true, y_pred, name, img):
        y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
        if np.isnan(y_pred).any():
            return {'image': img, 'model': name, 'R2': -999, 'RMSE': 999, 'MAE': 999, 'MRE(%)': 999}
        mre = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1e-6, y_true))) * 100
        return {'image': img, 'model': name, 'R2': r2_score(y_true, y_pred),
                'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
                'MAE': mean_absolute_error(y_true, y_pred), 'MRE(%)': mre}
    @staticmethod
    def shap_analysis(model, X, feature_names, out_dir, img_name, device):
        if not HAS_SHAP: return
        model.eval()
        def pred_fn(x_np):
            xt = torch.from_numpy(x_np.astype(np.float32)).to(device)
            with torch.no_grad(): out = model(xt)
            return out.cpu().numpy().ravel()
        bg_size = min(100, X.shape[0]); bg_indices = np.random.choice(X.shape[0], bg_size, replace=False)
        bg_data = X[bg_indices]; test_size = min(500, X.shape[0])
        test_indices = np.random.choice(X.shape[0], test_size, replace=False); test_data = X[test_indices]
        expl = shap.KernelExplainer(pred_fn, bg_data); sv = expl.shap_values(test_data)
        meanabs = np.mean(np.abs(sv), axis=0)
        IO.save_csv({'feature': feature_names, 'shap_mean_abs': meanabs.tolist()}, out_dir, f"shap_{img_name}")

class GatedLinearUnit(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.proj = nn.Linear(in_features, out_features * 2)
    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * torch.sigmoid(gate)

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            GatedLinearUnit(dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return x + self.block(x)

class AdvancedMLP(nn.Module):
    def __init__(self, total_dim, hidden_dims=(256, 128, 64), dropout=0.2):
        super().__init__()
        self.initial_layer = nn.Linear(total_dim, hidden_dims[0])
        self.res_blocks = nn.ModuleList([
            ResidualBlock(dim, dropout) for dim in hidden_dims
        ])
        self.linear_layers = nn.ModuleList([
            nn.Linear(hidden_dims[i], hidden_dims[i+1]) for i in range(len(hidden_dims) - 1)
        ])
        self.output_layer = nn.Linear(hidden_dims[-1], 1)

    def forward(self, x):
        x = self.initial_layer(x)
        for i in range(len(self.res_blocks)):
            x = self.res_blocks[i](x)
            if i < len(self.linear_layers):
                x = self.linear_layers[i](x)
        return self.output_layer(x).view(-1)

class NumpyDataset(Dataset):
    def __init__(self, X, y): self.X = X.astype(np.float32); self.y = y.astype(np.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return torch.from_numpy(self.X[i]), torch.tensor(self.y[i], dtype=torch.float32)

def _extract_features_for_point(r, c, feature_cube):
    k_half = CFG.NEIGHBORHOOD_SIZE // 2
    r_start = max(0, r - k_half)
    r_end = min(feature_cube.shape[1], r + k_half + 1)
    c_start = max(0, c - k_half)
    c_end = min(feature_cube.shape[2], c + k_half + 1)
    window = feature_cube[:, r_start:r_end, c_start:c_end]
    means = np.mean(window, axis=(1, 2))
    stds = np.std(window, axis=(1, 2))
    mins = np.min(window, axis=(1, 2))
    maxs = np.max(window, axis=(1, 2))
    return np.hstack([means, stds, mins, maxs])

def assemble_meta_features(oof_dict, base_maps, raw_map, sample_rows, sample_cols, used_bases):
    all_feature_maps = [raw_map] + [np.expand_dims(base_maps[b], axis=0) for b in used_bases]
    feature_cube = np.vstack(all_feature_maps)
    
    print("正在并行提取邻域特征...")
    results = Parallel(n_jobs=-1)(
        delayed(_extract_features_for_point)(r, c, feature_cube) 
        for r, c in tqdm(zip(sample_rows, sample_cols), total=len(sample_rows))
    )
    neighborhood_features = np.array(results)

    point_features = np.column_stack([oof_dict[b] for b in used_bases])
    X_meta = np.hstack([point_features, neighborhood_features])
    
    feature_names = [f'oof_{b}' for b in used_bases]
    map_names = [f'band_{i}' for i in range(raw_map.shape[0])] + [f'base_map_{b}' for b in used_bases]
    for stat in ['mean', 'std', 'min', 'max']:
        feature_names.extend([f'hood_{stat}_{name}' for name in map_names])

    return X_meta.astype(np.float32), feature_names

def predict_full_map_mlp(model, base_maps, raw_map, scaler, device, used_bases):
    nrows, ncols = raw_map.shape[1], raw_map.shape[2]
    all_feature_maps = [raw_map] + [np.expand_dims(base_maps[b], axis=0) for b in used_bases]
    feature_cube = np.vstack(all_feature_maps)
    
    all_rows, all_cols = np.meshgrid(np.arange(nrows), np.arange(ncols), indexing='ij')
    all_coords = list(zip(all_rows.flatten(), all_cols.flatten()))
    
    final_map_flat = np.zeros(nrows * ncols, dtype=np.float32)

    for i in tqdm(range(0, len(all_coords), CFG.PREDICT_BATCH_SIZE), desc="Predicting Full Map"):
        batch_coords = all_coords[i:i+CFG.PREDICT_BATCH_SIZE]
        batch_rows, batch_cols = zip(*batch_coords)

        batch_neighborhood_features = Parallel(n_jobs=-1)(
            delayed(_extract_features_for_point)(r, c, feature_cube) for r, c in batch_coords
        )
        batch_neighborhood_features = np.array(batch_neighborhood_features)

        batch_point_features_list = []
        for b in used_bases:
            batch_point_features_list.append(base_maps[b][batch_rows, batch_cols])
        batch_point_features = np.column_stack(batch_point_features_list)
        
        X_batch = np.hstack([batch_point_features, batch_neighborhood_features])
        X_batch_std = scaler.transform(X_batch)
        
        model.eval()
        with torch.no_grad():
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == 'cuda')):
                preds = model(torch.from_numpy(X_batch_std).float().to(device)).cpu().numpy()
        
        indices = np.ravel_multi_index((batch_rows, batch_cols), (nrows, ncols))
        final_map_flat[indices] = preds

    return final_map_flat.reshape(nrows, ncols)

def interpolate_negatives(pred_map):
    corrected_map = np.copy(pred_map)
    bad_mask = corrected_map < 0
    
    if not np.any(bad_mask):
        return corrected_map
        
    print(f"检测到 {np.sum(bad_mask)} 个负值预测点，正在进行griddata插值修正...")
    
    good_mask = ~bad_mask
    good_coords = np.argwhere(good_mask)
    bad_coords = np.argwhere(bad_mask)
    good_values = corrected_map[good_mask]
    
    if len(good_values) < 3:
        print("  -> 有效点过少，无法进行插值，将直接截断为0。")
        corrected_map[bad_mask] = 0
        return corrected_map
    
    interpolated_values = griddata(good_coords, good_values, bad_coords, method='linear')
    
    nan_mask = np.isnan(interpolated_values)
    if np.any(nan_mask):
        print(f"  -> {np.sum(nan_mask)} 个点无法线性插值，将使用最近邻填充。")
        nearest_values = griddata(good_coords, good_values, bad_coords[nan_mask], method='nearest')
        interpolated_values[nan_mask] = nearest_values

    corrected_map[bad_mask] = interpolated_values
    np.clip(corrected_map, 0, None, out=corrected_map)
    print("插值修正完成。")
    return corrected_map

def process_single_image(img_path, pts_path, out_dir, enable_gpu=True):
    os.makedirs(out_dir, exist_ok=True); img_name = os.path.basename(img_path).replace('.tif',''); metrics = []
    gt, proj, ncols, nrows, data, bands_count, nodata = IO.validate_image(img_path); pts = pd.read_csv(pts_path)
    if 'Longitude' not in pts.columns and pts.shape[1] >= 3: pts.columns = ['Longitude','Latitude','Depth'] + pts.columns.tolist()[3:]
    pts = pts.rename(columns={c:c for c in pts.columns}); lons, lats, depths = pts['Longitude'].values, pts['Latitude'].values, pts['depth'].values
    xy_proj = IO.lonlat_to_proj(pts, proj)
    rows, cols, depths, lons, lats, dists = match_points(xy_proj, gt, ncols, nrows, lons, lats, depths, out_dir, img_name)
    if len(depths)==0: print(f"{img_name} 无有效样本，跳过"); return metrics
    
    X_samples = data[:, rows, cols].T
    flat_spec = data.transpose(1,2,0).reshape(-1, bands_count)
    scaler = StandardScaler(); X_scaled = scaler.fit_transform(X_samples)
    if 'SVR' in ['RandomForest','XGBoost','SVR','CatBoost']: flat_scaled = scaler.transform(flat_spec)
    
    base_names = ['RandomForest','XGBoost','SVR','CatBoost']
    oof_dict, base_maps = {}, {}
    for name in base_names:
        print(f"\n--- 基础模型: {name} ---"); search_X = X_scaled if name=='SVR' else X_samples
        try: best = Searcher.search_base(name, search_X, depths) 
        except Exception as e: print(e); best = {}
        current_flat_spec = flat_scaled if name=='SVR' else flat_spec
        oof_pred, fmap_pred, _ = BaseTrainer.train_oof(name, X_samples, depths, current_flat_spec, nrows, ncols, best, scaler if name=='SVR' else None)
        if fmap_pred is None: continue
        oof_dict[name] = oof_pred; base_maps[name] = fmap_pred
        metrics.append(MetaHelper.eval(depths, oof_pred, f"Base_{name}", img_name))
        IO.save_tiff(fmap_pred, gt, proj, out_dir, f"depth_base_{name}_{img_name}")
        IO.save_csv({'lon': lons, 'lat': lats, 'depth_obs': depths, 'depth_pred': oof_pred}, out_dir, f'oof_pred_base_{name}_{img_name}')

    used = [n for n in base_names if n in base_maps]
    if len(used)==0: print("无可用基础模型"); return metrics
    
    X_meta, feature_names = assemble_meta_features(oof_dict, base_maps, data, rows, cols, used)
    print(f"邻域特征提取完毕。X_meta 形状: {X_meta.shape}")

    # <--- 核心修正3: 添加NaN检查 ---
    if np.isnan(X_meta).any():
        print("警告: 特征矩阵中包含NaN值，将用0填充。")
        X_meta = np.nan_to_num(X_meta, nan=0.0, posinf=0.0, neginf=0.0)

    X_train_meta, X_val_meta, y_train, y_val = train_test_split(X_meta, depths, test_size=0.2, random_state=CFG.RANDOM_STATE)
    
    meta_scaler = StandardScaler()
    X_train = meta_scaler.fit_transform(X_train_meta)
    X_val = meta_scaler.transform(X_val_meta)
    
    joblib.dump(meta_scaler, os.path.join(out_dir, f'meta_scaler_{img_name}.joblib'))
    
    device = torch.device('cuda' if enable_gpu and torch.cuda.is_available() else 'cpu')
    
    train_loader = DataLoader(NumpyDataset(X_train, y_train), batch_size=CFG.BATCH_SIZE, shuffle=True, num_workers=CFG.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(NumpyDataset(X_val, y_val), batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=CFG.NUM_WORKERS, pin_memory=True)
    
    print("\n--- 训练AdvancedMLP元模型 (已加速) ---")
    meta_model = AdvancedMLP(total_dim=X_train.shape[1], dropout=CFG.MLP_DROPOUT).to(device)
    
    optimizer = optim.AdamW(meta_model.parameters(), lr=CFG.MLP_LR, weight_decay=CFG.WEIGHT_DECAY, fused=(device.type == 'cuda'))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.MLP_EPOCHS_MAX, eta_min=1e-6)
    loss_fn = nn.MSELoss()
    
    scaler = torch.amp.GradScaler(enabled=(device.type == 'cuda'))
    
    # <--- 核心修正1: 更稳健的最佳模型保存逻辑 ---
    best_loss = float('inf'); patience_counter = 0
    best_model_state = None

    for epoch in range(CFG.MLP_EPOCHS_MAX):
        meta_model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == 'cuda')):
                preds = meta_model(xb)
                loss = loss_fn(preds, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        meta_model.eval(); val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == 'cuda')):
                    preds = meta_model(xb)
                    loss_val = loss_fn(preds, yb)
                    if not torch.isnan(loss_val):
                         val_loss += loss_val.item() * xb.size(0)
        
        val_loss /= len(y_val)
        scheduler.step()
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}: Val Loss: {val_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
            
        if val_loss < best_loss:
            best_loss = val_loss; patience_counter = 0
            best_model_state = copy.deepcopy(meta_model.state_dict())
            torch.save(best_model_state, os.path.join(out_dir, f"mlp_meta_model_{img_name}.pth"))
        else:
            patience_counter += 1
            if patience_counter >= CFG.EARLY_STOP:
                print(f"Early stopping at epoch {epoch+1}")
                break

    if best_model_state is None:
        print("警告: 训练不稳定，未能保存任何有效模型。")
        return metrics # 提前退出
        
    meta_model.load_state_dict(best_model_state)
    
    meta_model.eval()
    with torch.no_grad():
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == 'cuda')):
            X_meta_std = meta_scaler.transform(X_meta)
            final_preds = meta_model(torch.from_numpy(X_meta_std).float().to(device)).cpu().numpy()
    
    final_preds_corrected = np.copy(final_preds)
    final_preds_corrected[final_preds_corrected < 0] = 0
    
    metrics.append(MetaHelper.eval(depths, final_preds_corrected, "Final_MLP_on_Samples", img_name))
    IO.save_csv({'lon': lons, 'lat': lats, 'depth_obs': depths, 'depth_pred': final_preds_corrected}, out_dir, f'final_pred_mlp_on_samples_{img_name}')
    
    if CFG.ENABLE_SHAP:
        print("正在进行SHAP分析...")
        MetaHelper.shap_analysis(meta_model, X_meta_std, feature_names, out_dir, img_name, device)
        print("SHAP分析完成。")
    else:
        print("已跳过SHAP分析。")
        
    print("\n--- MLP全图预测 ---")
    final_map_raw = predict_full_map_mlp(meta_model, base_maps, data, meta_scaler, device, used)
    final_map_corrected = interpolate_negatives(final_map_raw)
    IO.save_tiff(final_map_corrected, gt, proj, out_dir, f"depth_final_mlp_interpolated_{img_name}")
    print("MLP全图预测及修正完成。")

    return metrics

def batch_process(image_folder, points_folder, output_folder, enable_gpu=True):
    os.makedirs(output_folder, exist_ok=True); imgs = [f for f in os.listdir(image_folder) if f.endswith('.tif')]
    all_metrics = []; t0 = time.time()
    for im in imgs:
        base = os.path.splitext(im)[0]; pts = os.path.join(points_folder, base + '.csv')
        if not os.path.exists(pts): print(f"缺样本 {base}.csv 跳过"); continue
        print('\n\n====================', im, '====================')
        m = process_single_image(os.path.join(image_folder, im), pts, output_folder, enable_gpu)
        all_metrics.extend(m)
    if all_metrics: IO.save_csv(all_metrics, output_folder, 'all_metrics_summary')
    print(f"Done. 用时 {(time.time()-t0)/60:.2f} 分钟")

if __name__ == '__main__':
#     # #100点结果
#     batch_process(
#         image_folder=r'D:\2article\tif\adg490',
#         points_folder=r'D:\2article\csv\训练csv改\100',
#         output_folder=r'D:\2article\result\全新特征\100结果\adg490',
#         enable_gpu=True
#     )


#     batch_process(
#         image_folder=r'D:\2article\tif\aphy490+adg490',
#         points_folder=r'D:\2article\csv\训练csv改\100',
#         output_folder=r'D:\2article\result\全新特征\100结果\aphy490+adg490',
#         enable_gpu=True
#     )

#     batch_process(
#         image_folder=r'D:\2article\tif\kd+adg490',
#         points_folder=r'D:\2article\csv\训练csv改\100',
#         output_folder=r'D:\2article\result\全新特征\100结果\kd+adg490',
#         enable_gpu=True
#     )

#     batch_process(
#         image_folder=r'D:\2article\tif\kd+aphy490',
#         points_folder=r'D:\2article\csv\训练csv改\100',
#         output_folder=r'D:\2article\result\全新特征\100结果\kd+aphy490',
#         enable_gpu=True    
#     )
#     batch_process(
#         image_folder=r'D:\2article\tif\kd',
#         points_folder=r'D:\2article\csv\训练csv改\100',
#         output_folder=r'D:\2article\result\全新特征\100结果\kd',
#         enable_gpu=True
#     )

#     batch_process(
#         image_folder=r'D:\2article\tif\aphy490',
#         points_folder=r'D:\2article\csv\训练csv改\100',
#         output_folder=r'D:\2article\result\全新特征\100结果\aphy490',
#         enable_gpu=True
#     )



#     batch_process(
#         image_folder=r'D:\2article\tif\styz',
#         points_folder=r'D:\2article\csv\训练csv改\100',
#         output_folder=r'D:\2article\result\全新特征\100结果\styz',
#         enable_gpu=True
#     )


#     batch_process(
#         image_folder=r'D:\2article\tif\ys',
#         points_folder=r'D:\2article\csv\训练csv改\100',
#         output_folder=r'D:\2article\result\全新特征\100结果\ys',
#         enable_gpu=True
#     )

# # 500点结果
#     batch_process(
#         image_folder=r'D:\2article\tif\adg490',
#         points_folder=r'D:\2article\csv\训练csv改\500',
#         output_folder=r'D:\2article\result\全新特征\500结果\adg490',
#         enable_gpu=True
#     )

#     batch_process(
#         image_folder=r'D:\2article\tif\aphy490+adg490',
#         points_folder=r'D:\2article\csv\训练csv改\500',
#         output_folder=r'D:\2article\result\全新特征\500结果\aphy490+adg490',
#         enable_gpu=True
#     )

#     batch_process(
#         image_folder=r'D:\2article\tif\kd+adg490',
#         points_folder=r'D:\2article\csv\训练csv改\500',
#         output_folder=r'D:\2article\result\全新特征\500结果\kd+adg490',
#         enable_gpu=True
#     )

#     batch_process(
#         image_folder=r'D:\2article\tif\kd',
#         points_folder=r'D:\2article\csv\训练csv改\500',
#         output_folder=r'D:\2article\result\全新特征\500结果\kd',
#         enable_gpu=True
#     )

#     batch_process(
#         image_folder=r'D:\2article\tif\aphy490',
#         points_folder=r'D:\2article\csv\训练csv改\500',
#         output_folder=r'D:\2article\result\全新特征\500结果\aphy490',
#         enable_gpu=True
#     )

#     batch_process(
#         image_folder=r'D:\2article\tif\kd+aphy490',
#         points_folder=r'D:\2article\csv\训练csv改\500',
#         output_folder=r'D:\2article\result\全新特征\500结果\kd+aphy490',
#         enable_gpu=True
#     )

#     batch_process(
#         image_folder=r'D:\2article\tif\styz',
#         points_folder=r'D:\2article\csv\训练csv改\500',
#         output_folder=r'D:\2article\result\全新特征\500结果\styz',
#         enable_gpu=True
#     )


#     batch_process(
#         image_folder=r'D:\2article\tif\ys',
#         points_folder=r'D:\2article\csv\训练csv改\500',
#         output_folder=r'D:\2article\result\全新特征\500结果\ys',
#         enable_gpu=True
#     )
# #1000点结果
#     batch_process(
#             image_folder=r'D:\2article\tif\adg490',
#         points_folder=r'D:\2article\csv\训练csv改\1000',
#         output_folder=r'D:\2article\result\全新特征\1000结果\adg490',
#         enable_gpu=True
#     )

#     batch_process(
#         image_folder=r'D:\2article\tif\aphy490+adg490',
#         points_folder=r'D:\2article\csv\训练csv改\1000',
#         output_folder=r'D:\2article\result\全新特征\1000结果\aphy490+adg490',
#         enable_gpu=True
#     )

#     batch_process(
#         image_folder=r'D:\2article\tif\kd+adg490',
#         points_folder=r'D:\2article\csv\训练csv改\1000',
#         output_folder=r'D:\2article\result\全新特征\1000结果\kd+adg490',
#         enable_gpu=True
#     )

#     batch_process(
#         image_folder=r'D:\2article\tif\kd',
#         points_folder=r'D:\2article\csv\训练csv改\1000',
#         output_folder=r'D:\2article\result\全新特征\1000结果\kd',
#         enable_gpu=True
#     )

#     batch_process(
#         image_folder=r'D:\2article\tif\aphy490',
#         points_folder=r'D:\2article\csv\训练csv改\1000',
#         output_folder=r'D:\2article\result\全新特征\1000结果\aphy490',
#         enable_gpu=True
#     )

#     batch_process(
#         image_folder=r'D:\2article\tif\kd+aphy490',
#         points_folder=r'D:\2article\csv\训练csv改\1000',
#         output_folder=r'D:\2article\result\全新特征\1000结果\kd+aphy490',
#         enable_gpu=True
#     )

#     batch_process(
#         image_folder=r'D:\2article\tif\styz',
#         points_folder=r'D:\2article\csv\训练csv改\1000',
#         output_folder=r'D:\2article\result\全新特征\1000结果\styz',
#         enable_gpu=True
#     )


#     batch_process(
#         image_folder=r'D:\2article\tif\ys',
#         points_folder=r'D:\2article\csv\训练csv改\1000',
#         output_folder=r'D:\2article\result\全新特征\1000结果\ys',
#         enable_gpu=True
#     )
# #2000点结果
#     batch_process(
#             image_folder=r'D:\2article\tif\adg490',
#         points_folder=r'D:\2article\csv\训练csv改\2000',
#         output_folder=r'D:\2article\result\全新特征\2000结果\adg490',
#         enable_gpu=True
#     )

#     batch_process(
#         image_folder=r'D:\2article\tif\aphy490+adg490',
#         points_folder=r'D:\2article\csv\训练csv改\2000',
#         output_folder=r'D:\2article\result\全新特征\2000结果\aphy490+adg490',
#         enable_gpu=True
#     )

#     batch_process(
#         image_folder=r'D:\2article\tif\kd+adg490',
#         points_folder=r'D:\2article\csv\训练csv改\2000',
#         output_folder=r'D:\2article\result\全新特征\2000结果\kd+adg490',
#         enable_gpu=True
#     )

#     batch_process(
#         image_folder=r'D:\2article\tif\kd',
#         points_folder=r'D:\2article\csv\训练csv改\2000',
#         output_folder=r'D:\2article\result\全新特征\2000结果\kd',
#         enable_gpu=True
#     )

#     batch_process(
#         image_folder=r'D:\2article\tif\aphy490',
#         points_folder=r'D:\2article\csv\训练csv改\2000',
#         output_folder=r'D:\2article\result\全新特征\2000结果\aphy490',
#         enable_gpu=True
#     )

#     batch_process(
#         image_folder=r'D:\2article\tif\kd+aphy490',
#         points_folder=r'D:\2article\csv\训练csv改\2000',
#         output_folder=r'D:\2article\result\全新特征\2000结果\kd+aphy490',
#         enable_gpu=True
#     )

#     batch_process(
#         image_folder=r'D:\2article\tif\styz',
#         points_folder=r'D:\2article\csv\训练csv改\2000',
#         output_folder=r'D:\2article\result\全新特征\2000结果\styz',
#         enable_gpu=True
#     )


#     batch_process(
#         image_folder=r'D:\2article\tif\ys',
#         points_folder=r'D:\2article\csv\训练csv改\2000',
#         output_folder=r'D:\2article\result\全新特征\2000结果\ys',
#         enable_gpu=True
#     )

# #3000点结果
    batch_process(
        image_folder=r'D:\2article\tif\adg490',
        points_folder=r'D:\2article\csv\训练csv改\3000',
        output_folder=r'D:\2article\result\全新特征\3000结果\adg490',
        enable_gpu=True
    )

    batch_process(
        image_folder=r'D:\2article\tif\aphy490+adg490',
        points_folder=r'D:\2article\csv\训练csv改\3000',
        output_folder=r'D:\2article\result\全新特征\3000结果\aphy490+adg490',
        enable_gpu=True
    )

    batch_process(
        image_folder=r'D:\2article\tif\kd+adg490',
        points_folder=r'D:\2article\csv\训练csv改\3000',
        output_folder=r'D:\2article\result\全新特征\3000结果\kd+adg490',
        enable_gpu=True
    )

    # batch_process(
    #     image_folder=r'D:\2article\tif\kd',
    #     points_folder=r'D:\2article\csv\训练csv改\3000',
    #     output_folder=r'D:\2article\result\全新特征\3000结果\kd',
    #     enable_gpu=True
    # )

    # batch_process(
    #     image_folder=r'D:\2article\tif\aphy490',
    #     points_folder=r'D:\2article\csv\训练csv改\3000',
    #     output_folder=r'D:\2article\result\全新特征\3000结果\aphy490',
    #     enable_gpu=True
    # )

    batch_process(
        image_folder=r'D:\2article\tif\kd+aphy490',
        points_folder=r'D:\2article\csv\训练csv改\3000',
        output_folder=r'D:\2article\result\全新特征\3000结果\kd+aphy490',
        enable_gpu=True
    )

    # batch_process(
    #     image_folder=r'D:\2article\tif\styz',
    #     points_folder=r'D:\2article\csv\训练csv改\3000',
    #     output_folder=r'D:\2article\result\全新特征\3000结果\styz',
    #     enable_gpu=True
    # )


    # batch_process(
    #     image_folder=r'D:\2article\tif\ys',
    #     points_folder=r'D:\2article\csv\训练csv改\3000',
    #     output_folder=r'D:\2article\result\全新特征\3000结果\ys',
    #     enable_gpu=True
    # )