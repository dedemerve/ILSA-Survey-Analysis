from src.analysis.corpus import build_frames, load_corpus


def test_corpus_has_212_unique_articles_and_keeps_doi_conflict():
    articles, warnings = load_corpus()
    assert len(articles) == 212
    assert any("DOI CONFLICT" in w and "Tin et al" in w for w in warnings)
    frames = build_frames(articles)
    assert frames["articles"]["article_id"].is_unique
