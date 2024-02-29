from huggingface_hub import snapshot_download

download_list = [
    "nomic-ai/nomic-embed-text-v1",
]

for repo in download_list:
    snapshot_download(
        repo_id=repo,
        local_dir="../pretrained/"
        + repo,
        cache_dir="../pretrained/"
        + repo,
        local_dir_use_symlinks=False,
        token="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"
    )
