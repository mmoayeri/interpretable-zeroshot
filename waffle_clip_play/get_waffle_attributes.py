import pickle as pkl
import numpy as np
from typing import List
import json

pre_descriptor_text = ""
label_before_text = "A photo of a "
descriptor_separator = ", "
label_after_text = "."


def wordify(string: str):
    word = string.replace("_", " ")
    return word


def load_json(filename: str):
    if not filename.endswith(".json"):
        filename += ".json"
    with open(filename, "r") as fp:
        return json.load(fp)


def make_descriptor_sentence(descriptor: str):
    if descriptor.startswith("a") or descriptor.startswith("an"):
        return f"which is {descriptor}"
    elif (
        descriptor.startswith("has")
        or descriptor.startswith("often")
        or descriptor.startswith("typically")
        or descriptor.startswith("may")
        or descriptor.startswith("can")
    ):
        return f"which {descriptor}"
    elif descriptor.startswith("used"):
        return f"which is {descriptor}"
    else:
        return f"which has {descriptor}"


def modify_descriptor(descriptor: str, apply_changes: bool):
    if apply_changes:
        return make_descriptor_sentence(descriptor)
    return descriptor


def get_waffle_descriptions(
    dataset: str = "food101",
    key_list: List[str] = ["dog", "cat"],
    waffle_count: int = 15,
):
    descriptor_fname = f"./descriptors/descriptors_{dataset}.json"
    gpt_descriptions = load_json(descriptor_fname)

    # Replace empty descriptor lists if necessary.
    gpt_descriptions = {
        key: item if len(item) else [""] for key, item in gpt_descriptions.items()
    }

    # Get complete list of available descriptions.
    descr_list = [list(values) for values in gpt_descriptions.values()]
    descr_list = np.array([x for y in descr_list for x in y])
    print("descr_list ", descr_list)

    # List of available classes.
    key_list = list(gpt_descriptions.keys())

    ### Descriptor Makers.
    structured_descriptor_builder = (
        lambda item, cls: f"{pre_descriptor_text}{label_before_text}{wordify(cls)}{descriptor_separator}{modify_descriptor(item, True)}{label_after_text}"
    )
    word_list = pkl.load(open("word_list.pkl", "rb"))

    avg_num_words = int(
        np.max([np.round(np.mean([len(wordify(x).split(" ")) for x in key_list])), 1])
    )
    avg_word_length = int(
        np.round(
            np.mean(
                [np.mean([len(y) for y in wordify(x).split(" ")]) for x in key_list]
            )
        )
    )
    word_list = [x[:avg_word_length] for x in word_list]

    # (Lazy solution) Extract list of available random characters from gpt description list. Ideally we utilize a separate list.
    character_list = [x.split(" ") for x in descr_list]
    character_list = [
        x.replace(",", "").replace(".", "")
        for x in np.unique([x for y in character_list for x in y])
    ]
    character_list = np.unique(list("".join(character_list)))

    num_spaces = (
        int(np.round(np.mean([np.sum(np.array(list(x)) == " ") for x in key_list]))) + 1
    )
    num_chars = int(
        np.ceil(np.mean([np.max([len(y) for y in x.split(" ")]) for x in key_list]))
    )

    num_chars += num_spaces - num_chars % num_spaces
    sample_key = ""

    for s in range(num_spaces):
        for _ in range(num_chars // num_spaces):
            sample_key += "a"
        if s < num_spaces - 1:
            sample_key += " "

    gpt_descriptions = {key: [] for key in gpt_descriptions.keys()}

    for key in key_list:
        for _ in range(waffle_count):
            base_word = ""
            for a in range(avg_num_words):
                base_word += np.random.choice(word_list, 1, replace=False)[0]
                if a < avg_num_words - 1:
                    base_word += " "
            gpt_descriptions[key].append(structured_descriptor_builder(base_word, key))
            noise_word = ""
            use_key = sample_key if len(key) >= len(sample_key) else key
            for c in sample_key:
                if c != " ":
                    noise_word += np.random.choice(character_list, 1, replace=False)[0]
                else:
                    noise_word += ", "
            gpt_descriptions[key].append(structured_descriptor_builder(noise_word, key))

    match_key = np.random.choice(key_list)
    gpt_descriptions = {key: gpt_descriptions[match_key] for key in key_list}
    for key in gpt_descriptions:
        gpt_descriptions[key] = [
            x.replace(wordify(match_key), wordify(key)) for x in gpt_descriptions[key]
        ]
    return get_waffle_descriptions


if __name__ == "__main__":
    print(get_waffle_descriptions())
