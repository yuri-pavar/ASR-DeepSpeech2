import warnings

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.datasets.data_utils import get_dataloaders
from src.trainer import Trainer
from src.utils.init_utils import set_random_seed, setup_saving_and_logging

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="baseline")
def main(config):
    """
    Main script for training. Instantiates the model, optimizer, scheduler,
    metrics, logger, writer, and dataloaders. Runs Trainer to train and
    evaluate the model.

    Args:
        config (DictConfig): hydra experiment config.
    """
    # config - словарь словарей конфигов
    set_random_seed(config.trainer.seed)
    # Преобразование config (тип DictConfig) в стандартный словарь с помощью 
    # OmegaConf.to_container гарантирует, что project_config будет стандартным
    #  словарем Python
    project_config = OmegaConf.to_container(config)
    logger = setup_saving_and_logging(config)
    # instantiate позволяет изменять параметры в конфигах "налету", 
    # не меняя каждый раз значения в конфгах вручную
    # optimizer:
    #   _target_: my_app.Optimizer
    #   algo: SGD
    #   lr: 0.01
    # # override parameters on the call-site
    # opt = instantiate(cfg.optimizer, lr=0.2)
    writer = instantiate(config.writer, logger, project_config)

    if config.trainer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.trainer.device

    # setup text_encoder
    text_encoder = instantiate(config.text_encoder)

    # setup data_loader instances
    # batch_transforms should be put on device
    dataloaders, batch_transforms = get_dataloaders(config, text_encoder, device)

    # build model architecture, then print to console
    model = instantiate(config.model, n_tokens=len(text_encoder)).to(device)
    logger.info(model)

    # get function handles of loss and metrics
    loss_function = instantiate(config.loss_function).to(device)

    metrics = {"train": [], "inference": []}
    for metric_type in ["train", "inference"]:
        for metric_config in config.metrics.get(metric_type, []):
            # use text_encoder in metrics
            metrics[metric_type].append(
                instantiate(metric_config, text_encoder=text_encoder)
            )

    # build optimizer, learning rate scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = instantiate(config.optimizer, params=trainable_params)
    lr_scheduler = instantiate(config.lr_scheduler, optimizer=optimizer)

    # epoch_len = number of iterations for iteration-based training
    # epoch_len = None or len(dataloader) for epoch-based training
    epoch_len = config.trainer.get("epoch_len")

    trainer = Trainer(
        model=model,
        criterion=loss_function,
        metrics=metrics,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        text_encoder=text_encoder,
        config=config,
        device=device,
        dataloaders=dataloaders,
        epoch_len=epoch_len,
        logger=logger,
        writer=writer,
        batch_transforms=batch_transforms,
        skip_oom=config.trainer.get("skip_oom", True),
    )

    trainer.train()


if __name__ == "__main__":
    main()
