import pytest
from modules.FeatureExtractor import FeatureExtractor

@pytest.fixture
def html_sample_data():
    return [
        {
            "description": "An epic <b>RPG</b> with <i>dragons</i>!",
            "price": 59.99,
            "metacritic": 85,
            "median_forever": 120
        },
        {
            "description": "<p>Fast-paced <strong>action</strong> shooter.</p>",
            "price": 19.99,
            "metacritic": 70,
            "median_forever": 50
        },
        {
            "description": "<div>Open world <span>adventure</span> game</div>",
            "price": 39.99,
            "metacritic": 75,
            "median_forever": 80
        }
    ]

@pytest.fixture
def feature_extractor():
    return FeatureExtractor()

def test_html_tag_removal(feature_extractor, html_sample_data):
    processed_texts = [feature_extractor.preprocess_text(game["description"]) for game in html_sample_data]
    expected_texts = [
        "an epic rpg with dragons",
        "fast-paced action shooter",
        "open world adventure game"
    ]
    assert processed_texts == expected_texts

def test_nested_html_tags(feature_extractor):
    nested_html = "<div>Game with <b>nested <i>tags</i></b></div>"
    processed = feature_extractor.preprocess_text(nested_html)
    assert processed == "game with nested tags"

def test_special_html_characters(feature_extractor):
    special_chars = "Game with &lt;special&gt; characters &amp; symbols"
    processed = feature_extractor.preprocess_text(special_chars)
    assert processed == "game with special characters symbols"

def test_empty_html_tags(feature_extractor):
    empty_tags = "<div></div><p></p>Empty content"
    processed = feature_extractor.preprocess_text(empty_tags)
    assert processed == "empty content"

def test_html_attributes(feature_extractor):
    with_attributes = '<div class="game" id="123">Game with attributes</div>'
    processed = feature_extractor.preprocess_text(with_attributes)
    assert processed == "game with attributes"

def test_multiple_spaces_after_html_removal(feature_extractor):
    multiple_spaces = "<div>Game</div>  <p>with</p>  <span>spaces</span>"
    processed = feature_extractor.preprocess_text(multiple_spaces)
    assert processed == "game with spaces" 