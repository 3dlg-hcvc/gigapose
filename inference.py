import os
import os.path as osp
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from einops import repeat
import pytorch_lightning as pl
from src.utils.logging import get_logger, log_image
from src.utils.batch import BatchedData, gather
from src.utils.optimizer import HybridOptim
from torchvision.utils import save_image
from src.utils.time import Timer
from src.models.loss import cosine_similarity
from src.lib3d.torch import (
    cosSin,
    get_relative_scale_inplane,
    geodesic_distance,
)
from src.libVis.torch import (
    plot_Kabsch,
    plot_keypoints_batch,
    save_tensor_to_image,
)
from src.models.poses import ObjectPoseRecovery
import src.megapose.utils.tensor_collection as tc
from src.utils.inout import save_predictions_from_batched_predictions

from src.custom_megapose.template_dataset import TemplateDataset, NearestTemplateFinder
import torch

logger = get_logger(__name__)


class TemplateSet(Dataset):
    def __init__(
        self,
        root_dir,
        dataset_name,
        template_config,
        transforms,
        **kwargs,
    ):
        self.root_dir = Path(root_dir)
        self.dataset_name = dataset_name
        self.transforms = transforms

        # load the template dataset
        self.model_infos = [{"obj_id": obj_id.strip()} for obj_id in open("/local-scratch/qiruiw/research/diorama/data/wss/wss_models.txt")]

        template_config.dir += f"/{dataset_name}"
        self.template_dataset = TemplateDataset.from_config(
            self.model_infos, template_config
        )
        self.template_finder = NearestTemplateFinder(template_config)

    def __len__(self):
        return len(self.model_infos)

    def __getitem__(self, index):
        # loading templates
        if "lmo" in self.dataset_name:
            label = LMO_index_to_ID[index]
        else:
            label = f"{index+1}"

        # load template data
        template_data = self.template_dataset.get_object_templates(label)
        data, poses = template_data.read_test_mode()

        # crop the template
        cropped_data = self.transforms.crop_transform(data["box"], images=data["rgba"])
        cropped_data["images"][:, :3] = self.transforms.normalize(
            cropped_data["images"][:, :3]
        )
        data["K"] = torch.from_numpy(self.template_dataset.K).float()

        out_data = tc.PandasTensorCollection(
            K=data["K"],
            rgb=cropped_data["images"][:, :3],
            mask=cropped_data["images"][:, -1],
            M=cropped_data["M"],
            poses=poses,
            infos=pd.DataFrame(),
        )
        return out_data


class GigaPose(pl.LightningModule):
    def __init__(
        self,
        model_name,
        ae_net,
        ist_net,
        training_loss,
        testing_metric,
        optim_config,
        log_interval,
        log_dir,
        max_num_dets_per_forward=None,
        **kwargs,
    ):
        # define the network
        super().__init__()
        self.model_name = model_name
        self.ae_net = ae_net
        self.ist_net = ist_net
        # self.training_loss = training_loss
        self.testing_metric = testing_metric

        self.max_num_dets_per_forward = max_num_dets_per_forward

        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(osp.join(self.log_dir, "predictions"), exist_ok=True)

        # for testing
        self.template_datas = {}
        self.pose_recovery = {}
        self.run_id = None
        self.template_datasets = None
        self.test_dataset_name = None

        logger.info("Initialize GigaPose done!")

    def validate_contrast_loss(self, batch, idx_batch, split):
        src_feat = self.ae_net(batch.src_img)
        tar_feat = self.ae_net(batch.tar_img)

        preds = self.testing_metric.val(
            src_feat=src_feat,
            tar_feat=tar_feat,
            src_mask=batch.src_mask,
            tar_mask=batch.tar_mask,
        )
        setattr(batch, "pred_src_pts", preds.src_pts)
        setattr(batch, "pred_tar_pts", preds.tar_pts)

    def validation_step(self, batch, idx_batch):
        _ = self.validate_contrast_loss(batch, idx_batch, "val")

    def set_template_data(self, dataset_name):
        logger.info("Initializing template data ...")
        template_dataset = self.template_datasets[dataset_name]
        names = ["rgb", "mask", "K", "M", "poses", "ae_features", "ist_features"]
        template_data = {name: BatchedData(None) for name in names}

        for idx in tqdm(range(len(template_dataset))):
            for name in names:
                if name in ["ae_features", "ist_features"]:
                    continue
                if name == "rgb":
                    templates = template_dataset[idx].rgb.to(self.device)
                    if self.max_num_dets_per_forward is None:
                        template_data[name].append(templates)

                    ae_features = self.ae_net(templates)
                    template_data["ae_features"].append(ae_features)

                    ist_features = self.ist_net.forward_by_chunk(templates)
                    template_data["ist_features"].append(ist_features)
                else:
                    tmp = getattr(template_dataset[idx], name)
                    template_data[name].append(tmp.to(self.device))
        if self.max_num_dets_per_forward is not None:
            names.remove("rgb")
        for name in names:
            template_data[name].stack()
            template_data[name] = template_data[name].data

        self.template_datas[dataset_name] = tc.PandasTensorCollection(
            infos=pd.DataFrame(), **template_data
        )
        self.pose_recovery[dataset_name] = ObjectPoseRecovery(
            template_K=template_data["K"],
            template_Ms=template_data["M"],
            template_poses=template_data["poses"],
        )
        # num_obj = len(template_data["K"])

    def eval_retrieval(
        self,
        batch,
        idx_batch,
        dataset_name,
        sort_pred_by_inliers=True,
    ):
        torch.cuda.empty_cache()
        # prepare template data
        if dataset_name not in self.template_datas:
            self.set_template_data(dataset_name)

        template_data = self.template_datas[dataset_name]
        pose_recovery = self.pose_recovery[dataset_name]
        times = {"neighbor_search": None, "final_step": None}

        B, C, H, W = batch.tar_img.shape
        device = batch.tar_img.device

        # if low_memory_mode, two detections are forward at a time
        list_idx_sample = []
        if self.max_num_dets_per_forward is not None:
            for start_idx in np.arange(0, B, self.max_num_dets_per_forward):
                end_idx = min(start_idx + self.max_num_dets_per_forward, B)
                idx_sample_ = torch.arange(start_idx, end_idx, device=device)
                list_idx_sample.append(idx_sample_)
        else:
            idx_sample = torch.arange(0, B, device=device)
            list_idx_sample.append(idx_sample)

        for idx_sub_batch, idx_sample in enumerate(list_idx_sample):
            # compute target features
            tar_ae_features = self.ae_net(batch.tar_img[idx_sample])
            tar_label_np = np.asarray(
                batch.infos.label[idx_sample.cpu().numpy()]
            ).astype(np.int32)
            tar_label = torch.from_numpy(tar_label_np).to(device)

            # template data
            src_ae_features = template_data.ae_features[tar_label - 1]
            src_masks = template_data.mask[tar_label - 1]

            # Step 1: Nearest neighbor search
            predictions_ = self.testing_metric.test(
                src_feats=src_ae_features,
                tar_feat=tar_ae_features,
                src_masks=src_masks,
                tar_mask=batch.tar_mask[idx_sample],
                max_batch_size=None,
            )
            predictions_.infos = batch.infos
            if idx_sub_batch == 0:
                predictions = predictions_
            else:
                predictions.cat_df(predictions_)

        # # calculate prediction
        # pred_poses = self.pose_recovery[dataset_name].forward_recovery(
        #     tar_label=tar_label,
        #     tar_K=batch.tar_K,
        #     tar_M=batch.tar_M,
        #     pred_src_views=predictions.id_src,
        #     pred_M=predictions.M.clone(),
        # )
        # predictions.register_tensor("pred_poses", pred_poses)

    @torch.no_grad()
    def test_step(self, batch, idx_batch):
        self.eval_retrieval(
            batch,
            idx_batch=idx_batch,
            dataset_name=self.test_dataset_name,
        )
        return 0
