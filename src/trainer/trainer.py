from pathlib import Path

import pandas as pd

from src.logger.utils import plot_spectrogram
from src.metrics.tracker import MetricTracker
from src.metrics.utils import calc_cer, calc_wer
from src.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.optimizer.zero_grad()

        outputs = self.model(**batch)
        # print('---- outputs ------')
        # print(outputs)
        batch.update(outputs)

        all_losses = self.criterion(**batch)
        batch.update(all_losses)

        if self.is_train:
            batch["loss"].backward()  # sum of all losses is always called loss
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            metrics.update(met.name, met(**batch))
        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        # method to log data from you batch
        # such as audio, text or images, for example

        # print(batch)

        # logging scheme might be different for different partitions
        if mode == "train":  # the method is called only every self.log_step steps
            self.log_spectrogram(**batch)
        else:
            # Log Stuff
            self.log_spectrogram(**batch)
            self.log_predictions(**batch)

    def log_spectrogram(self, spectrogram, **batch):
        spectrogram_for_plot = spectrogram[0].detach().cpu()
        # spectrogram_for_plot = spectrogram[0].detach().cpu().numpy()

        # print(f'spectrogram TYPE - {type(spectrogram)}')
        # print(f'spectrogram SHAPE - {spectrogram.shape}')
        # print(f'spectrogram_for_plot TYPE - {type(spectrogram_for_plot)}')
        # print(f'spectrogram_for_plot SHAPE - {spectrogram_for_plot.shape}')

        image = plot_spectrogram(spectrogram_for_plot)
        # print(f'image TYPE - {type(image)}')
        # print(f'image SHAPE - {image.shape}')
        self.writer.add_image("spectrogram", image)

    
    # def log_predictions(
    #     self, text, log_probs, log_probs_length, audio_path, examples_to_log=10, **batch
    # ):
    def log_predictions(
        self, text, log_probs, log_probs_length, audio_path, audio, examples_to_log=5, **batch
    ):
        # TODO add beam search
        # Note: by improving text encoder and metrics design
        # this logging can also be improved significantly

        # print(f'txt = {text}')
        # print(f'log_probs= {log_probs}')
        # print(f'log_probs_length= {log_probs_length}')
        # print(f'audio_path= {audio_path}')
        # print(f'audio= {audio}')

        log_probs = log_probs.detach().cpu().numpy()
        log_probs_lengths = log_probs_length.detach().cpu().numpy()

        # argmax_inds = log_probs.cpu().argmax(-1).numpy()
        argmax_inds = log_probs.argmax(-1)
        # print(f'argmax_inds1 = {argmax_inds}')
        argmax_inds = [
            inds[: int(ind_len)]
            for inds, ind_len in zip(argmax_inds, log_probs_length.numpy())
        ]
        # print(f'argmax_inds2 = {argmax_inds}')
        argmax_texts_raw = [self.text_encoder.decode(inds) for inds in argmax_inds]
        argmax_texts = [self.text_encoder.ctc_decode(inds) for inds in argmax_inds]
        # bs_texts = [
        #     self.text_encoder.ctc_beam_search(proba[:length], 3)
        #     for proba, length in zip(log_probs, log_probs_lengths)
        # ]
        # print(f'argmax_texts_raw = {argmax_texts_raw}')
        # print(f'argmax_texts= {argmax_texts}')
        # tuples = list(zip(argmax_texts, text, argmax_texts_raw, audio_path))
        tuples = list(zip(
            argmax_texts, 
            # bs_texts
            text, 
            argmax_texts_raw, 
            audio_path,
            audio
            ))

        rows = {}
        for (
            pred, 
            target, 
            raw_pred, 
            audio_path,
            audio_aug
        ) in tuples[:examples_to_log]:
            target = self.text_encoder.normalize_text(target)
            wer = calc_wer(target, pred) * 100
            cer = calc_cer(target, pred) * 100

            rows[Path(audio_path).name] = {
                "target": target,
                "raw prediction": raw_pred,
                "predictions": pred,
                "wer": wer,
                "cer": cer,
                # "audio_aug": self.writer.cometml.Audio(audio_aug, sample_rate=16000),

            }
        self.writer.add_audio(
            "audio_aug", audio_aug, sample_rate=16000
        )
        self.writer.add_table(
            "predictions", pd.DataFrame.from_dict(rows, orient="index")
        )
