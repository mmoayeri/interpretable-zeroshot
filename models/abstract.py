from abc import ABC, abstractmethod


class AttrAwareClassifier(ABC):
    '''
    Main modular part here: making a final class prediction based off subpopulation similarities

    Questions: do our baselines need to adhere to this structure?
    '''

    def __init__(self, image_encoder, text_encoder, dataset):

    def forward(self, x):

        return pred_label, dict_of_other_things

    def compute_image_features(self, dataset, cache_path):
        # pass dataset through image encoder

        # cache dict w/ image features, class labels, and attribute labels

    def compute_subpop_sims(self, attrs_by_class, cache_path):
        # Obtain and embed text descriptions of each subpopulation

        # Obtain (or load cached) image features

        # Compute cosine similarities


    @abstractmethod
    def predict(self, subpop_sims):
        # Return class given similarities to each subpopulation
        
