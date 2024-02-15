from investments.timeseries.main import main as run_models
from investments.outputs.main import main as run_outputs
from investments.preprocess.main import main as run_preprocess


def main():
    run_preprocess()
    run_models()
    run_outputs()


if __name__ == "__main__":
    main()
