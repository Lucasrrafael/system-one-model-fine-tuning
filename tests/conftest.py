from pathlib import Path

import pytest

CLEF_REPO = "Cloudflare/clef-flash"
# Mesma revisão do configs/clef-colab.env.
CLEF_REVISION = "fb4417d31c1d0f9e7e77f1141b40aaede7491e00"
CLEF_SMALL_FILES = ["joint_schema_model.py", "config.json", "joint_head_config.json", "tokenizer.json",
                    "tokenizer_config.json", "chat_template.jinja", "processor_config.json"]


@pytest.fixture(scope="session")
def clef_files() -> Path:
    """Arquivos pequenos do clef-flash (código, tokenizer, configs), sem os pesos de 19 GB."""
    from huggingface_hub import hf_hub_download

    cache = Path(__file__).parent / ".cache" / "clef-flash"
    try:
        for name in CLEF_SMALL_FILES:
            hf_hub_download(CLEF_REPO, name, revision=CLEF_REVISION, local_dir=cache)
    except Exception as e:  # sem rede
        pytest.skip(f"sem acesso ao Hugging Face: {e}")
    return cache
