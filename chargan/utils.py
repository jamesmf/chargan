def prime():
    """
    Use this function to prompt the pre-download of the default feature extraction
    model during image build time.
    """
    from transformers import pipeline

    primer = pipeline("feature-extraction")
    result = primer(["hello"])
