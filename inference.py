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
from src.utils.batch import BatchedData, gather
from src.utils.optimizer import HybridOptim
from torchvision.utils import save_image
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
from src.utils.pil import open_image

from src.custom_megapose.template_dataset import NearestTemplateFinder
from src.custom_megapose.transform import Transform, ScaleTransform

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from torch.utils.data import DataLoader
from src.utils.logging import start_disable_output, stop_disable_output


@dataclass
class TemplateData:
    label: str
    template_dir: str
    num_templates: int
    TWO_init: Transform
    pose_path: Optional[str] = None

    @staticmethod
    def from_dict(template_gt) -> "TemplateData":
        assert isinstance(template_gt, dict)
        data = TemplateData(
            label=template_gt["label"],
            template_dir=str(template_gt["template_dir"]),
            pose_path=str(template_gt["pose_path"]),
            num_templates=int(template_gt["num_templates"]),
            TWO_init=ScaleTransform(scale_factor=template_gt["scale_factor"]),
        )
        return data

    def load_template(self, view_id, inplane=None):
        image_path = f"{self.template_dir}/{view_id}.png"
        rgba = open_image(image_path, inplane)
        box = rgba.getbbox()
        box_size = (box[2] - box[0], box[3] - box[1])
        if min(box_size) == 0:
            box = (0, 0, int(rgba.size[0]), int(rgba.size[1]))
            print(f"Template {image_path} has zero area, setting to null template")
        return {"rgba": np.array(rgba), "box": np.array(box)}

    def load_set_of_templates(self, view_ids, reload=False, inplanes=None, reset=True):
        if inplanes is None:
            inplanes = [None for _ in view_ids]
        root_dir = os.path.dirname(self.template_dir)
        obj_id = os.path.basename(self.template_dir)

        preprocessed_file = f"{root_dir}/preprocessed/{int(obj_id):06d}.npz"
        if os.path.exists(preprocessed_file) and reload and not reset:
            data = np.load(preprocessed_file)
            rgba = torch.from_numpy(data["rgba"]).float()
            box = torch.from_numpy(data["box"]).long()
            return {"rgba": rgba, "box": box}
        else:
            os.makedirs(f"{root_dir}/preprocessed", exist_ok=True)
            data = {"rgba": [], "box": []}
            for view_id, inplane in zip(view_ids, inplanes):
                view_data = self.load_template(view_id, inplane=inplane)
                rgba = torch.from_numpy(view_data["rgba"] / 255).float()
                box = torch.from_numpy(view_data["box"]).long()
                data["rgba"].append(rgba)
                data["box"].append(box)
            data["rgba"] = torch.stack(data["rgba"]).permute(0, 3, 1, 2)
            data["box"] = torch.stack(data["box"])
            if reload:
                np.savez(
                    preprocessed_file,
                    rgba=data["rgba"].numpy(),
                    box=data["box"].numpy(),
                )
        return data

    def apply_transform(self, transform, data):
        data["rgba"], data["M"] = transform(
            images=data["rgba"], boxes=data["box"], return_transform=True
        )
        return data

    def load_pose(self, view_ids=None, inplanes=[0]):
        poses = np.load(self.pose_path)
        if view_ids is None:  # all poses for testing mode
            poses = [Transform(poses[i]) * self.TWO_init for i in range(len(poses))]
            return torch.stack([pose.toTensor() for pose in poses])
        else:  # only load poses for training mode
            inplane_transforms = [Transform.from_inplane(inp) for inp in inplanes]
            poses = [
                inplane_transforms[i] * Transform(poses[view_ids[i]]) * self.TWO_init
                for i in range(len(view_ids))
            ]
            return poses

    def read_test_mode(self):
        data = self.load_set_of_templates(view_ids=np.arange(0, self.num_templates))
        poses = self.load_pose()
        return data, poses


@dataclass
class TemplateDataset:
    def __init__(
        self,
        object_templates: List[TemplateData],
    ):
        self.list_object_templates = object_templates
        self.label_to_objects = {obj.label: obj for obj in object_templates}
        self.K = np.array(
            [193.9897, 0.0, 112, 0.0, 193.9897, 112, 0.0, 0.0, 1.0]
        ).reshape((3, 3))

    def __getitem__(self, idx: int) -> TemplateData:
        return self.list_object_templates[idx]

    def get_object_templates(self, label: str) -> TemplateData:
        return self.label_to_objects[label]

    def __len__(self) -> int:
        return len(self.list_object_templates)

    @property
    def objects(self) -> List[TemplateData]:
        """Returns a list of objects in this dataset."""
        return self.list_object_templates

    def from_config(model_infos) -> "TemplateDataset":
        template_datas = []
        for model_info in tqdm(model_infos):
            obj_id = model_info["obj_id"]
            template_metaData = {"label": str(obj_id)}
            template_metaData["num_templates"] = 24
            template_metaData["template_dir"] = f"/local-scratch/qiruiw/research/diorama/data/wss-neutral-renders/{obj_id}"
            template_metaData["pose_path"] = "/local-scratch/qiruiw/research/diorama/data/obj_poses.npy"
            template_metaData["scale_factor"] = 1
            template_data = TemplateData.from_dict(template_metaData)
            template_datas.append(template_data)
        return TemplateDataset(template_datas)


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
        self.model_infos = [{"obj_id": obj_id.strip()} for obj_id in open("/project/3dlg-hcvc/diorama/wss/wss_models.txt")]

        template_config.dir += f"/{dataset_name}"
        self.template_dataset = TemplateDataset.from_config(self.model_infos)
        self.template_finder = NearestTemplateFinder(template_config)

    def __len__(self):
        return len(self.model_infos)

    def __getitem__(self, index):
        # load template data
        template_data = self.template_dataset.get_object_templates(self.model_infos[index]["obj_id"])
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
        testing_metric,
        log_dir,
        max_num_dets_per_forward=None,
        **kwargs,
    ):
        # define the network
        super().__init__()
        self.model_name = model_name
        self.ae_net = ae_net
        self.ist_net = ist_net
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

    def encode_multiviews(self, dataset_name):
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
            self.encode_multiviews(dataset_name)

        template_data = self.template_datas[dataset_name]
        pose_recovery = self.pose_recovery[dataset_name]

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



@hydra.main(version_base=None, config_path="configs", config_name="test")
def run_test(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    # cfg_trainer = cfg.machine.trainer
    os.makedirs(cfg.save_dir, exist_ok=True)

    # trainer = instantiate(cfg_trainer)
    cfg.model._target_ = "__main__.GigaPose"
    model = instantiate(cfg.model)

    # cfg.data.test.dataloader.dataset_name = cfg.test_dataset_name
    # cfg.data.test.dataloader.batch_size = cfg.machine.batch_size
    # cfg.data.test.dataloader.load_gt = False
    # test_dataset = instantiate(cfg.data.test.dataloader)
    # test_dataloader = DataLoader(
    #     test_dataset.web_dataloader.datapipeline,
    #     batch_size=1,  # a single image may have multiples instances
    #     num_workers=cfg.machine.num_workers,
    #     collate_fn=test_dataset.collate_fn,
    # )

    # set template dataset as a part of the model
    cfg.data.test.dataloader.dataset_name = cfg.test_dataset_name
    cfg.data.test.dataloader._target_ = "__main__.TemplateSet"
    template_dataset = instantiate(cfg.data.test.dataloader)
    import pdb; pdb.set_trace()

    model.template_datasets = {cfg.test_dataset_name: template_dataset}
    model.test_dataset_name = cfg.test_dataset_name
    model.max_num_dets_per_forward = cfg.max_num_dets_per_forward
    
    model.encode_multiviews(cfg.test_dataset_name)

    # model.log_interval = len(test_dataloader) // 30

    # trainer.test(
    #     model, dataloaders=test_dataloader, ckpt_path=cfg.model.checkpoint_path
    # )


if __name__ == "__main__":
    run_test()
