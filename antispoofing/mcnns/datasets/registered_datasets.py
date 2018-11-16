# -*- coding: utf-8 -*-

from antispoofing.mcnns.datasets.livdetiris17_nd import LivDetIris17_ND
from antispoofing.mcnns.datasets.ndcld15 import  NDCLD15
from antispoofing.mcnns.datasets.NDContactLensesDataset import NDContactLensesDataset_15
from antispoofing.mcnns.datasets.ndcontactlenses import NDContactLenses
from antispoofing.mcnns.datasets.NDSpoofingPreClassification import NDSpoofingPreClassification
from antispoofing.mcnns.datasets.livdetiris17_warsaw import LivDetIris17_Warsaw
from antispoofing.mcnns.datasets.livdet_nd_ww import LivDet_ND_WW
from antispoofing.mcnns.datasets.livdet_nd_val import LivDetND
from antispoofing.mcnns.datasets.livdet_ww_val import LivDetWW
from antispoofing.mcnns.datasets.livdet_nd_ww_val import LivDetNDWW
from antispoofing.mcnns.datasets.livdet_clarkson_val import LivDetClarkson
from antispoofing.mcnns.datasets.livdet_nd_ww_cl import LivDetNDWWCL
from antispoofing.mcnns.datasets.livdet_iiitd_val import LivDetIIITD
from antispoofing.mcnns.datasets.livdet_combined import LivDetCombined

registered_datasets = {0: LivDetIris17_ND,
                       1: NDCLD15,
                       2: NDContactLenses,
                       3: NDContactLensesDataset_15,
                       4: NDSpoofingPreClassification,
                       5: LivDetIris17_Warsaw,
                       6: LivDet_ND_WW,
                       7: LivDetND,
                       8: LivDetWW,
                       9: LivDetNDWW,
                       10: LivDetClarkson,
                       11: LivDetNDWWCL,
                       12: LivDetIIITD,
                       13: LivDetCombined
                       }
