# copyright CEA-LIST/DIASI/SIALV/LVA (2022)
# author CEA-LIST/DIASI/SIALV/LVA <julien.denize@cea.fr>
# license CeCILL version 2.1

from torch.optim.lr_scheduler import MultiStepLR

from sce.schedulers.linear_warmup_cosine_annealing_lr import LinearWarmupCosineAnnealingLR
from sce.schedulers.scheduler_factory import scheduler_factory
