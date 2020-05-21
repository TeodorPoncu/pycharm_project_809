import dataset
from torch.utils.data import DataLoader
from src.trainers.stack_trainer_v8 import StackTrainer
import yaml
from argparse import Namespace
import wandb


if __name__ == '__main__':
    with open('config.yml') as f:
        cfg = yaml.load(f)
        wandb.init(config=cfg, project="image-synthesis")
        cfg = Namespace(**cfg)
    data_set = dataset.MismatchTextToImageDataset(cfg)
    data_loarder = DataLoader(data_set, num_workers=8, batch_size=cfg.batch_size, shuffle=True)
    model = StackTrainer(cfg)
    init_done = False
    total_iter = 0
    for _ in range(cfg.num_epochs * 10):
        for idx, item in enumerate(data_loarder):
            total_iter = total_iter + cfg.batch_size
            model.set_input(item)
            if not init_done:
                init_done = True
                model.init_trainer_network()
            model.optimize()

            if idx % cfg.loss_display_freq == 0:
                wandb.log(model.get_current_losses(), step=total_iter)

            if idx % cfg.visual_display_freq == 0:
                wandb.log({'results': wandb.Image(model.get_current_visuals(), caption=model.text_input[0])}, step=total_iter)