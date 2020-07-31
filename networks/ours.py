import torch.nn.functional as F
from torch import nn

from networks.cpvton import GMM, TOM, load_checkpoint


class UnifiedCpVton(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.gmm = GMM(opt)
        self.tom = TOM(opt)

    def load_pretrained(self, gmm_checkpoint_path=None, tom_checkpoint_path=None):
        if gmm_checkpoint_path:
            load_checkpoint(self.gmm, gmm_checkpoint_path)
        if tom_checkpoint_path:
            load_checkpoint(self.tom, tom_checkpoint_path)

    def forward(self, agnostic, cloth):
        warped_grid, theta = self.gmm(agnostic, cloth)
        warped_cloth = F.grid_sample(cloth, warped_grid, padding_mode="border")

        p_rendered, m_composite, p_tryon = self.tom(agnostic, warped_cloth)

        return {
            "warped_grid": warped_grid,
            "warped_cloth": warped_cloth,
            "m_composite": m_composite,
            "p_rendered": p_rendered,
            "p_tryon": p_tryon,
        }
