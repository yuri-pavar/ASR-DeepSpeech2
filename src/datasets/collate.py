import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    # instance_data = {
    #         "audio": audio,
    #         "spectrogram": spectrogram,
    #         "text": text,
    #         "text_encoded": text_encoded,
    #         "audio_path": audio_path,
    #     }

    # pass  # TODO
    audio_list = []
    spectrogram_list = []
    text_list = []
    text_encoded_list = []
    audio_path_list = []

    text_encoded_lengths_list = []
    spectrogram_lengths_list = []
    

    for item in dataset_items:
        audio = item["audio"].squeeze(0)
        audio_list.append(audio)

        spectrogram = item["spectrogram"].squeeze(0).transpose(0, -1)
        spectrogram_list.append(spectrogram)

        text_list.append(item["text"])

        text_encoded = item["text_encoded"].squeeze(0).transpose(0, -1)
        text_encoded_list.append(text_encoded)

        audio_path_list.append(item["audio_path"])

        text_encoded_lengths_list.append(item["text_encoded"].shape[1])

        spectrogram_lengths_list.append(item["spectrogram"].shape[2])


    audio_list = pad_sequence(audio_list, batch_first=True, padding_value=0.0).transpose(1, -1)
    spectrogram_list = pad_sequence(spectrogram_list, batch_first=True, padding_value=0.0).transpose(1, -1)
    text_encoded_list = pad_sequence(text_encoded_list, batch_first=True, padding_value=0).transpose(1, -1)
        
    text_encoded_lengths_list = torch.tensor(text_encoded_lengths_list, dtype=torch.long)
    spectrogram_lengths_list = torch.tensor(spectrogram_lengths_list, dtype=torch.long)

    # print(f'spectrogram_list = {spectrogram_list.shape}')

    result_batch = {
        # "audio": audio_list,
        # "spectrogram": spectrogram_list,
        # "text": text_list,
        # "text_encoded": text_encoded_list,
        # "audio_path": audio_path_list,
        # "text_encoded_length": text_encoded_lengths_list,
        # "spectrogram_length": spectrogram_lengths_list

        "text_encoded_length": text_encoded_lengths_list,
        "spectrogram_length": spectrogram_lengths_list,
        "spectrogram": spectrogram_list,
        "text_encoded": text_encoded_list,
        "text": text_list,
        "audio": audio_list,
        "audio_path": audio_path_list,
    }

    return result_batch
