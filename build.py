# build.py
from . import config
from .common import write_to_standard_dataset
from .adapters.ragbench_adapter import load_ragbench_subset, ragbench_to_golden_drafts
from .adapters.msmarco_adapter import load_msmarco, msmarco_to_golden_drafts
from .adapters.halubench_adapter import load_halubench, halubench_to_golden_drafts

def build_standard_dataset():
    all_drafts = []

    for subset in config.RAGBENCH_SUBSETS:
        ds = load_ragbench_subset(subset)
        all_drafts += ragbench_to_golden_drafts(subset, ds, config.RAGBENCH_SAMPLES_PER_SUBSET)

    all_drafts += msmarco_to_golden_drafts(load_msmarco(), config.MSMARCO_SAMPLES)
    all_drafts += halubench_to_golden_drafts(load_halubench(), config.HALUBENCH_PASS_SAMPLES)

    write_to_standard_dataset(all_drafts, config.STANDARD_DATASET_PATH)

if __name__ == "__main__":
    build_standard_dataset()