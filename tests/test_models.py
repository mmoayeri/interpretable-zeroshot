"""
Tests basic model implementations
"""
from models.vlm import BLIP2, InstructBLIP
from PIL import Image
import requests
import pytest


class TestBLIPModels:
    IMAGE_URL: str = "https://raw.githubusercontent.com/salesforce/LAVIS/main/docs/_static/Confusing-Pictures.jpg"
    RAW_IMAGE: Image = Image.open(requests.get(IMAGE_URL, stream=True).raw).convert(
        "RGB"
    )
    DEVICE = "cpu"

    @pytest.mark.parametrize(
        "model", [BLIP2(device=DEVICE), InstructBLIP(device=DEVICE)]
    )
    def test_image_feature_extraction(self, model):
        image = (
            model.vis_processors["eval"](self.RAW_IMAGE).unsqueeze(0).to(self.DEVICE)
        )

        image_features = model.encode_image_batch(image)
        assert image_features.shape == (1, 32, 768)

    def test_blip2_text_feature_extraction(self):
        device = "cpu"
        blip2 = BLIP2(device=device)

        text_features = blip2.encode_texts(["a dog"], ["what is "])

        assert text_features.shape == (1, 4, 768)

    def test_blip2_projected_feature_extraction(self):
        device = "cpu"
        blip2 = BLIP2(device=device)
        image = blip2.vis_processors["eval"](self.RAW_IMAGE).unsqueeze(0).to(device)

        image_features = blip2.encode_image_batch(image, project_embeddings=True)
        assert image_features.shape == (1, 32, 256)

        text_features = blip2.encode_texts(
            ["a dog"], ["what is "], project_embeddings=True
        )
        assert text_features.shape == (1, 4, 256)
